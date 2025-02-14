use clap::Parser;
use llr_reco::LlrReconParams;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

#[derive(Debug, Parser)]
struct Args {
    /// path to recon parameters file to write
    parameter_file: PathBuf,
}


fn main() {
    let args = Args::parse();
    let params = LlrReconParams::default();
    let mut toml_string = toml::to_string(&params).expect("Can't serialize");
    toml_string.push('\n');
    let filename = args.parameter_file.with_extension("toml");
    let mut f = File::create(&filename).expect("Can't create file");
    f.write_all(toml_string.as_bytes()).expect("Can't write file");
    println!("wrote default parameter file to {}", filename.display());
}