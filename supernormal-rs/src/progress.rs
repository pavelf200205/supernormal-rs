//! Dependency-free terminal progress. Redirection gets occasional plain lines.
use std::{
    collections::VecDeque,
    io::{self, IsTerminal, Write},
    time::{Duration, Instant},
};

pub struct TrainingProgress {
    total: usize,
    started: Instant,
    last_draw: Instant,
    terminal: bool,
    width: usize,
    durations: VecDeque<f64>,
    window_seconds: f64,
}

impl TrainingProgress {
    pub fn new(total: usize) -> Self {
        Self {
            total,
            started: Instant::now(),
            last_draw: Instant::now() - Duration::from_secs(1),
            terminal: io::stderr().is_terminal(),
            width: 0,
            durations: VecDeque::new(),
            window_seconds: 0.0,
        }
    }

    pub fn update(&mut self, step: usize, seconds: f64, loss: f64, report: bool) {
        // Use several iterations to smooth occupancy updates and pipelined GPU
        // work. Checkpoint/mesh export time is excluded from training throughput.
        self.durations.push_back(seconds.max(1e-9));
        self.window_seconds += seconds.max(1e-9);
        while self.durations.len() > 128 || (self.durations.len() > 8 && self.window_seconds > 5.0)
        {
            self.window_seconds -= self.durations.pop_front().unwrap();
        }
        let done = step >= self.total;
        if self.terminal {
            if !done && self.last_draw.elapsed() < Duration::from_millis(250) {
                return;
            }
        } else if !report && !done {
            return;
        }
        let rate = self.durations.len() as f64 / self.window_seconds.max(1e-9);
        let fraction = (step as f64 / self.total.max(1) as f64).clamp(0.0, 1.0);
        let filled = (fraction * 12.0) as usize;
        let bar = format!("{}{}", "=".repeat(filled), "-".repeat(12 - filled));
        let eta = self.total.saturating_sub(step) as f64 / rate;
        let text = format!(
            "{:5.1}% [{}] {}/{} | {:.2} it/s | elapsed {} | ETA {} | loss {:.5}",
            fraction * 100.0,
            bar,
            step,
            self.total,
            rate,
            clock(self.started.elapsed().as_secs_f64()),
            clock(eta),
            loss,
        );
        let mut stderr = io::stderr().lock();
        if self.terminal {
            // Carriage return and space padding also work in Windows terminals
            // without relying on ANSI escape support.
            let padding = self.width.saturating_sub(text.len());
            let _ = write!(stderr, "\r{text}{}", " ".repeat(padding));
            let _ = stderr.flush();
            self.width = text.len();
        } else {
            let _ = writeln!(stderr, "{text}");
        }
        self.last_draw = Instant::now();
        if done && self.terminal {
            let _ = writeln!(stderr);
            self.width = 0;
        }
    }

    /// Make room for checkpoint/mesh messages or explicit profiling output.
    pub fn clear(&mut self) {
        if self.terminal && self.width > 0 {
            let mut stderr = io::stderr().lock();
            let _ = write!(stderr, "\r{}\r", " ".repeat(self.width));
            let _ = stderr.flush();
            self.width = 0;
        }
        self.last_draw = Instant::now() - Duration::from_secs(1);
    }
}

impl Drop for TrainingProgress {
    fn drop(&mut self) {
        // Leave errors on a fresh line if training exits between redraws.
        if self.terminal && self.width > 0 {
            let _ = writeln!(io::stderr());
        }
    }
}

fn clock(seconds: f64) -> String {
    let seconds = seconds.max(0.0) as u64;
    if seconds >= 3600 {
        format!(
            "{}:{:02}:{:02}",
            seconds / 3600,
            seconds / 60 % 60,
            seconds % 60
        )
    } else {
        format!("{:02}:{:02}", seconds / 60, seconds % 60)
    }
}
