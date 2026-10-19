use anyhow::Result;
use burn::tensor::{Device, DeviceKind, Tensor};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
mod data;
mod field;
mod kernels;
mod mc_table;
mod mesh;
mod render;
mod train;

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    cpu: bool,
    /// Prefer an integrated GPU (for example Intel UHD), instead of the default adapter.
    #[arg(long, conflicts_with = "cpu")]
    integrated: bool,
    #[arg(long, default_value = "auto", value_parser = ["auto", "vulkan", "dx12", "metal"])]
    api: String,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Doctor,
    Verify {
        model: PathBuf,
    },
    VerifyStep {
        model: PathBuf,
        fixture: PathBuf,
    },
    Train(train::TrainArgs),
    Mesh {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 256)]
        resolution: usize,
        #[arg(long, default_value_t = 8192)]
        query: usize,
        #[arg(long)]
        data: Option<PathBuf>,
    },
    VerifyRender {
        fixture: PathBuf,
        #[arg(long)]
        data: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = if cli.cpu {
        Device::flex()
    } else {
        use burn::tensor::wgpu::WgpuBackend;
        let api = match cli.api.as_str() {
            "vulkan" => WgpuBackend::Vulkan,
            "dx12" => WgpuBackend::Dx12,
            "metal" => WgpuBackend::Metal,
            _ => WgpuBackend::Auto,
        };
        Device::wgpu_options()
            .device_kind(if cli.integrated {
                DeviceKind::IntegratedGpu(0)
            } else {
                DeviceKind::DefaultDevice
            })
            .graphics_api(api)
            .init()
            .map_err(|e| anyhow::anyhow!("GPU initialization failed: {e:?}"))?
    }
    .autodiff();
    println!("Device: {:?}", device.identity());
    match cli.command {
        Command::Verify { model } => return field::verify(&model, device),
        Command::VerifyStep { model, fixture } => {
            return render::verify_step(&model, &fixture, device);
        }
        Command::Train(args) => return train::train(args, device),
        Command::Mesh {
            model,
            out,
            resolution,
            query,
            data,
        } => {
            let field = field::Field::load(&model, device)?;
            let (scale, offset) = if let Some(path) = data {
                let dataset = data::Dataset::load(&path)?;
                (dataset.scale, dataset.offset)
            } else {
                (1., [0.; 3])
            };
            return mesh::extract(&field, &out, resolution, query, scale, offset);
        }
        Command::VerifyRender { fixture, data } => {
            return render::verify(&fixture, data.as_deref());
        }
        Command::Doctor => {
            println!("Running hash grid forward GPU kernel parity test...");
            kernels::test_hash_grid_forward_parity(&device)?;

            println!("Running hash grid backward GPU kernel parity test...");
            kernels::test_hash_grid_backward_parity(&device)?;

            if std::path::Path::new("experiments/field_30k").exists() {
                println!("Testing full 14-level field features GPU vs CPU on real field_30k...");
                let field = field::Field::load(
                    std::path::Path::new("experiments/field_30k"),
                    device.clone(),
                )?;
                let pts = vec![
                    [0.123f32, -0.456, 0.789],
                    [-0.500, 0.250, -0.125],
                    [0.0, 0.0, 0.0],
                    [0.8, -0.8, 0.3],
                ];
                let (cpu_feat, _, _) = field.compute_features_impl(&pts, false);
                let xyz = burn::tensor::Tensor::from_data(
                    burn::tensor::TensorData::new(
                        pts.iter().flatten().copied().collect::<Vec<f32>>(),
                        [pts.len(), 3],
                    ),
                    &device,
                );
                let gpu_feat = field.compute_features_gpu(xyz);
                let cpu_vec: Vec<f32> = cpu_feat.try_into_vec_as::<f32>()?;
                let gpu_vec: Vec<f32> = gpu_feat.try_into_vec_as::<f32>()?;
                let mut max_diff = 0f32;
                for i in 0..pts.len() {
                    for c in 0..31 {
                        let c_val = cpu_vec[i * 31 + c];
                        let g_val = gpu_vec[i * 31 + c];
                        let diff = (c_val - g_val).abs();
                        if diff > max_diff {
                            max_diff = diff;
                        }
                        if diff > 1e-4 {
                            println!(
                                "  pt {i} chan {c}: CPU={c_val:.6} GPU={g_val:.6} diff={diff:.2e}"
                            );
                        }
                    }
                }
                println!("Full 14-level features max diff across all channels: {max_diff:.2e}");
            }
        }
    }
    let x = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device).require_grad();
    let loss = (x.clone() * x.clone()).sum();
    let grads = loss.backward();
    println!(
        "Gradient smoke test: {:?}",
        x.grad(&grads).unwrap().try_into_vec_as::<f32>()?
    );
    Ok(())
}
