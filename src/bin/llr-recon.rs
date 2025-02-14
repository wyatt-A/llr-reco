use clap::Parser;
use llr_reco::{llr_recon_exec, DataSetParams, LlrReconParams};
use std::fs::File;
use std::io::Read;

fn main() {
    let ds_params = DataSetParams::parse();
    match File::open(ds_params.recon_params.with_extension("toml")) {
        Ok(mut file) => {
            let mut toml_str = String::new();
            file.read_to_string(&mut toml_str).expect("Failed to read toml file");
            match toml::from_str::<LlrReconParams>(&toml_str) {
                Ok(recon_params) => {
                    llr_recon_exec(&ds_params, &recon_params);
                }
                Err(e) => {
                    panic!("failed to parse recon params file with error: {}", e);
                }
            }
        }
        Err(e) => {
            panic!("failed to open config file: {} with error {}", &ds_params.recon_params.display(), e);
        }
    }
}