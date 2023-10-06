use std::time::{Duration, Instant};

use ndarray::prelude::*;
use ndarray_rand::rand::prelude::*;
use ndarray_rand::{rand_distr::Uniform, RandomExt};

// one layer fully connected network
// with softmax loss and stochastic gradient descent
pub fn train(
    x: Array2<f32>,
    y: Array1<u8>,
    num_iterations: u32,
    reg: f32,
    step_size: f32,
    batch_size: usize,
) -> (Array2<f32>, Array2<f32>, Vec<f32>) {
    let start = Instant::now();

    let mut w = Array::random((28 * 28, 10), Uniform::new(0.0, 0.01));
    let mut b = Array::<f32, _>::zeros((1, 10));

    let mut r = Array1::range(0.0, x.shape()[0] as f32, 1.0);
    let mut x_batch = Array::<f32, _>::zeros((batch_size, 28 * 28));
    let mut y_batch = Array::<u8, _>::zeros((batch_size,));

    let mut losses = vec![];

    let num_examples = batch_size as f32;

    let mut rng = ndarray_rand::rand::thread_rng();

    let initialized_time = start.elapsed();

    let mut rng_times = vec![];
    let mut b_assign_times = vec![];
    let mut score_times = vec![];
    let mut loss_times = vec![];
    let mut gradient_times = vec![];
    let mut backprop_times = vec![];
    let mut iter_times = vec![];

    let mut then: Instant;

    for i in 1..=num_iterations {
        let iter_then = Instant::now();

        // batch creation for SDG
        then = Instant::now();
        r.as_slice_mut().unwrap().shuffle(&mut rng);
        rng_times.push(then.elapsed());

        then = Instant::now();
        for i in 0..batch_size {
            let index = r[i] as usize;
            x_batch.slice_mut(s![i, ..]).assign(&x.row(index));
            y_batch[i] = y[index];
        }
        b_assign_times.push(then.elapsed());

        // calculate the probs
        then = Instant::now();

        // (batch_size x 28*28) @ (28*28 x 10) = (batch_size x 10)
        let scores = (x_batch.dot(&w) + &b).map(|a| a.exp());
        let scores = &scores / &scores.sum_axis(Axis(1)).to_shape((batch_size, 1)).unwrap();
        score_times.push(then.elapsed());

        then = Instant::now();
        let iter = Array1::range(0.0, num_examples, 1.0);

        // loss value
        // (batch_size,)
        let correct_logprobs = iter
            .map(|&a| scores[[a as usize, y_batch[a as usize] as usize]])
            .map(|p| -p.ln());
        let loss = correct_logprobs.sum() / num_examples;
        losses.push(loss);
        // loss += 0.5 * reg * (&w * &w).sum(); // regularization
        loss_times.push(then.elapsed());

        if i % 10 == 0 {
            println!("Loss at iteration {} is {}", i, loss);
        }

        // gradients
        then = Instant::now();
        // (batch_size x 10)
        let mut dscores = scores.clone();
        iter.map(|&a| dscores[[a as usize, y_batch[a as usize] as usize]] -= 1.0);
        dscores = dscores.map(|d| d / num_examples);
        gradient_times.push(then.elapsed());

        // backpropagation
        then = Instant::now();
        // (28*28 x batch_size) @ (batch_size x 10) = (28*28 x 10)
        let dw = x_batch.t().dot(&dscores) + reg * &w;
        //     }
        // }
        // (10,)
        let db = dscores.sum_axis(Axis(0));
        w = w + (dw * -step_size);
        b = b + (db * -step_size);
        backprop_times.push(then.elapsed());

        iter_times.push(iter_then.elapsed());
    }

    rng_times.sort();
    b_assign_times.sort();
    score_times.sort();
    loss_times.sort();
    gradient_times.sort();
    backprop_times.sort();
    iter_times.sort();

    println!("Initialization time: {:?}", initialized_time);
    println!(
        "Rng times: worst={:?}, median={:?}, average={:?}, best={:?}",
        rng_times.last().unwrap(),
        rng_times[rng_times.len() / 2],
        rng_times.iter().sum::<Duration>() / rng_times.len() as u32,
        rng_times.first().unwrap(),
    );
    println!(
        "Batch assign times: worst={:?}, median={:?}, average={:?}, best={:?}",
        b_assign_times.last().unwrap(),
        b_assign_times[b_assign_times.len() / 2],
        b_assign_times.iter().sum::<Duration>() / b_assign_times.len() as u32,
        b_assign_times.first().unwrap(),
    );
    println!(
        "Score times: worst={:?}, median={:?}, average={:?}, best={:?}",
        score_times.last().unwrap(),
        score_times[score_times.len() / 2],
        score_times.iter().sum::<Duration>() / score_times.len() as u32,
        score_times.first().unwrap(),
    );
    println!(
        "Loss times: worst={:?}, median={:?}, average={:?}, best={:?}",
        loss_times.last().unwrap(),
        loss_times[loss_times.len() / 2],
        loss_times.iter().sum::<Duration>() / loss_times.len() as u32,
        loss_times.first().unwrap(),
    );
    println!(
        "Gradient times: worst={:?}, median={:?}, average={:?}, best={:?}",
        gradient_times.last().unwrap(),
        gradient_times[gradient_times.len() / 2],
        gradient_times.iter().sum::<Duration>() / gradient_times.len() as u32,
        gradient_times.first().unwrap(),
    );
    println!(
        "Backprop times: worst={:?}, median={:?}, average={:?}, best={:?}",
        backprop_times.last().unwrap(),
        backprop_times[backprop_times.len() / 2],
        backprop_times.iter().sum::<Duration>() / backprop_times.len() as u32,
        backprop_times.first().unwrap(),
    );
    println!(
        "Per-iteration times: worst={:?}, median={:?}, average={:?}, best={:?}",
        iter_times.last().unwrap(),
        iter_times[iter_times.len() / 2],
        iter_times.iter().sum::<Duration>() / iter_times.len() as u32,
        iter_times.first().unwrap(),
    );

    (w, b, losses)
}
