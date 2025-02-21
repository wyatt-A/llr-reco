use clap::Parser;
use llr_reco::diffusion_models::{calc_scalars, CalcScalarsArgs};

fn main() {
    let args = CalcScalarsArgs::parse();
    calc_scalars(&args);
}