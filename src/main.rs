use std::time::Instant;

use mnist::{MnistBuilder, NormalizedMnist};
use ndarray::prelude::*;

mod cpu;
mod gpu;

fn main() {
    let mut args = std::env::args();
    let _ = args.next().unwrap();
    // let base_path = args.next().unwrap();
    let implementation = args.next().unwrap_or("cpu".to_string());

    let NormalizedMnist {
        trn_img,    // (50000, 28, 28)
        trn_lbl,    // (50000, 1)
        val_img,    // (10000, 28, 28)
        val_lbl,    // (10000, 1)
        tst_img: _, // (10000, 28, 28)
        tst_lbl: _, // (10000, 1)
    } = MnistBuilder::new()
        // .base_path(&base_path)
        .label_format_digit()
        .training_set_length(50000)
        .validation_set_length(10000)
        .test_set_length(10000)
        .finalize()
        .normalize();

    let trn_img = Array2::from_shape_vec((50000, 28 * 28), trn_img).unwrap();
    let trn_lbl = Array1::from_shape_vec(50000, trn_lbl).unwrap();
    let val_img = Array2::from_shape_vec((10000, 28 * 28), val_img).unwrap();
    let val_lbl = Array1::from_shape_vec(10000, val_lbl).unwrap();
    let total = Instant::now();
    let (w, b, _losses) = match implementation.as_str() {
        "cpu" => cpu::train(trn_img, trn_lbl, 5000, 0.1, 0.01, 256),
        "gpu" => gpu::train(trn_img, trn_lbl.map(|&a| a as u32), 5000, 0.1, 0.01),
        other => panic!("Unrecognized implementation {:?}", other),
    };
    println!("Total elapsed time: {:?}", total.elapsed());

    // println!("Losses: {:?}", losses);
    let acc = accuracy(w, b, val_img, val_lbl);
    println!("Validation accuracy: {:?}", acc);
}

fn accuracy(w: Array2<f32>, b: Array2<f32>, x: Array2<f32>, y: Array1<u8>) -> f32 {
    let scores = x.dot(&w) + &b;
    let maxs = scores.map_axis(Axis(1), |a| {
        a.indexed_iter()
            .max_by(|(_, m), (_, n)| m.partial_cmp(n).unwrap())
            .unwrap()
            .0 as u8
    });
    maxs.iter()
        .zip(y.iter())
        .map(|(a, b)| (a == b) as usize)
        .sum::<usize>() as f32
        / maxs.len() as f32
}
