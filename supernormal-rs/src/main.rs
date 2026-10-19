use anyhow::Result;
use burn::tensor::{Device, DeviceKind, Tensor};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
mod data;
mod field;
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
        Command::Doctor => {}
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
