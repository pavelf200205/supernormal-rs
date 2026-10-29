//! Optional kernel timestamp windows and copy/work counters. Disabled by default.
//! Profiling changes dispatch behavior; use short checkpoint resumes, not normal
//! training throughput, to attribute individual kernels. Missing timing is not zero.
use cubecl::client::{Client, ProfileWindow};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

static ENABLED: AtomicBool = AtomicBool::new(false);
const WINDOWS_PER_LABEL: usize = 128;
#[derive(Default)]
struct State {
    counters: BTreeMap<String, u64>,
    windows: BTreeMap<String, usize>,
    pending: Vec<Pending>,
    errors: Vec<String>,
    address_reuse: Vec<serde_json::Value>,
}
struct Pending {
    label: &'static str,
    duration: cubecl::profile::ProfileDuration,
}
fn state() -> &'static Mutex<State> {
    static S: OnceLock<Mutex<State>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(State::default()))
}
pub fn configure(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
    if enabled {
        *state().lock().unwrap() = State::default();
        eprintln!(
            "Kernel diagnostics enabled (up to 128 windows per label). Profiling perturbs dispatch; first-use costs may be included."
        );
    }
}
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn count(label: &'static str, amount: usize) {
    if ENABLED.load(Ordering::Relaxed) {
        *state()
            .lock()
            .unwrap()
            .counters
            .entry(label.into())
            .or_default() += amount as u64;
    }
}
pub struct Span {
    client: Client,
    label: &'static str,
    window: Option<ProfileWindow>,
}
pub fn start(client: &Client, label: &'static str) -> Option<Span> {
    if !ENABLED.load(Ordering::Relaxed) {
        return None;
    }
    {
        let mut s = state().lock().unwrap();
        let n = s.windows.entry(label.into()).or_default();
        if *n >= WINDOWS_PER_LABEL {
            return None;
        }
        *n += 1;
    }
    match client.profile_start() {
        Ok(window) => Some(Span {
            client: client.clone(),
            label,
            window: Some(window),
        }),
        Err(e) => {
            state()
                .lock()
                .unwrap()
                .errors
                .push(format!("{label}: start: {e:?}"));
            None
        }
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        let Some(window) = self.window.take() else {
            return;
        };
        if std::thread::panicking() {
            self.client.profile_abandon(window);
            return;
        }
        match self.client.profile_end(window) {
            Ok(duration) => state().lock().unwrap().pending.push(Pending {
                label: self.label,
                duration,
            }),
            Err(e) => state()
                .lock()
                .unwrap()
                .errors
                .push(format!("{}: end: {e:?}", self.label)),
        }
    }
}
#[derive(Default, Serialize)]
struct Row {
    calls: usize,
    total_ms: f64,
    first_ms: f64,
    min_ms: f64,
    max_ms: f64,
    after_first_mean_ms: Option<f64>,
}
pub fn report(path: &Path) -> anyhow::Result<()> {
    if !ENABLED.load(Ordering::Relaxed) {
        return Ok(());
    }
    let (pending, counters, mut errors, address_reuse) = {
        let mut s = state().lock().unwrap();
        (
            std::mem::take(&mut s.pending),
            std::mem::take(&mut s.counters),
            std::mem::take(&mut s.errors),
            std::mem::take(&mut s.address_reuse),
        )
    };
    let mut rows: BTreeMap<String, Row> = BTreeMap::new();
    for p in pending {
        let method = p.duration.timing_method().to_string();
        match cubecl::future::block_on(p.duration.resolve()) {
            Some(ticks) => {
                let ms = ticks.duration().as_secs_f64() * 1000.;
                let row = rows.entry(format!("{} ({method})", p.label)).or_default();
                if row.calls == 0 {
                    row.first_ms = ms;
                    row.min_ms = ms;
                }
                row.calls += 1;
                row.total_ms += ms;
                row.min_ms = row.min_ms.min(ms);
                row.max_ms = row.max_ms.max(ms);
                if row.calls > 1 {
                    row.after_first_mean_ms =
                        Some((row.total_ms - row.first_ms) / (row.calls - 1) as f64);
                }
            }
            None => errors.push(format!("{}: no timestamp measurement returned", p.label)),
        }
    }
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "profiling_perturbs_execution": true, "windows_per_label_limit": WINDOWS_PER_LABEL,
            "timing_method_is_per_row": true, "counter_values_are_not_measured_DRAM_traffic": true,
            "kernels": rows, "counters": counters, "unavailable_or_failed_measurements": errors,
        "address_reuse_samples": address_reuse,
        }))?,
    )?;
    eprintln!("Kernel diagnostics: {}", path.display());
    Ok(())
}

/// Address-only model from at most 64 four-point groups once per level count.
/// Diagnostic-only readbacks intentionally perturb execution. These estimates
/// do not measure executed atomic instructions or account for gradient cancellation.
pub fn sample_addresses(
    points: &burn::tensor::Tensor<2>,
    adjoints: &burn::tensor::Tensor<2>,
    valid: usize,
    active: usize,
) {
    use std::sync::atomic::AtomicU32;
    static SAMPLED: AtomicU32 = AtomicU32::new(0);
    if !enabled() || valid == 0 {
        return;
    }
    let bit = 1u32 << active.min(14);
    if SAMPLED.fetch_or(bit, Ordering::Relaxed) & bit != 0 {
        return;
    }
    let n = valid.min(256);
    let xyz = points.clone().slice([0..n, 0..3]).try_into_vec_as::<f32>();
    let ds = adjoints
        .clone()
        .slice([0..n, 0..1])
        .try_into_vec_as::<f32>();
    let (xyz, ds) = match (xyz, ds) {
        (Ok(x), Ok(d)) => (x, d),
        (x, d) => {
            state().lock().unwrap().errors.push(format!(
                "address sampling failed: {:?}; {:?}",
                x.err(),
                d.err()
            ));
            return;
        }
    };
    let mut rows = Vec::new();
    for level in 0..active.min(14) {
        let res = (32f64 * 1.3195079107728942f64.powi(level as i32)).floor() as u32;
        let r1 = res + 1;
        let dense = r1.pow(3) <= 524288;
        let mut candidate = 0usize;
        let mut same_corner = 0usize;
        let mut all_corners = 0usize;
        for group in (0..n).step_by(4) {
            let mut per_corner: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
            let mut all = Vec::new();
            for point in group..(group + 4).min(n) {
                if ds[point] == 0. {
                    continue;
                }
                let base: [u32; 3] = std::array::from_fn(|d| {
                    (((xyz[point * 3 + d] + 1.) * 0.5).clamp(0., 1. - 1e-7) * res as f32).floor()
                        as u32
                });
                for corner in 0..8usize {
                    let x = base[0] + (corner as u32 & 1);
                    let y = base[1] + ((corner as u32 >> 1) & 1);
                    let z = base[2] + ((corner as u32 >> 2) & 1);
                    let idx = if dense {
                        x + y * r1 + z * r1 * r1
                    } else {
                        (x ^ y.wrapping_mul(2654435761) ^ z.wrapping_mul(805459861)) & 524287
                    };
                    candidate += 1;
                    if !per_corner[corner].contains(&idx) {
                        per_corner[corner].push(idx);
                    }
                    if !all.contains(&idx) {
                        all.push(idx);
                    }
                }
            }
            same_corner += per_corner.iter().map(Vec::len).sum::<usize>();
            all_corners += all.len();
        }
        rows.push(
            serde_json::json!({"level": level, "candidate_vertex_updates": candidate,
            "unique_same_corner_updates": same_corner, "unique_cross_corner_updates": all_corners}),
        );
    }
    state().lock().unwrap().address_reuse.push(serde_json::json!({
        "active_levels": active, "prefix_points": n, "sample_is_not_random": true,
        "model_only_not_GPU_atomic_counts": true, "both_channels_share_vertex_address": true,
        "zero_adjoints_excluded_but_feature_gradient_cancellation_not_modeled": true, "levels": rows,
    }));
}
