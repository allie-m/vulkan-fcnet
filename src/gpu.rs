// bind group (mutable storage):
// - weight array (28x28x10)
// - bias array (10)
// bind group (mutable storage):
// - x batch (batch_size x 10)
// - y batch (batch_size)
// push constants:
// - batch size
// - regularization
// - step size

// compute the random shuffles on the CPU and write them out to a buffer
// pass 1 : (do cpuside buffer copy instead?) fill x and y batches with stuff indexed from x/y by random numbers
// pass 2 : scores
// pass 3 : loss value
// pass 4 : gradient
// pass 5 : backprop

use ash::extensions::ext::DebugUtils;
use ash::vk;
use ash::vk::DebugUtilsMessengerEXT;
use ash::Device;
use ash::Entry;
use ash::Instance;
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::*;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use std::ffi::CStr;
use std::io::Cursor;
use std::mem::size_of;
use std::time::Duration;
use std::time::Instant;

pub fn train(
    x: Array2<f32>,
    y: Array1<u32>,
    num_iterations: u32,
    reg: f32,
    step_size: f32,
    // batch_size: usize,
) -> (Array2<f32>, Array2<f32>, Vec<f32>) {
    // this value is hardcoded into some of the shaders
    let batch_size = 256;

    let initialized_time = Instant::now();

    let init_weights = Array::random((28 * 28, 10), Uniform::new(0.0, 0.01));
    let mut r = Array::range(0.0, x.shape()[0] as f32, 1.0);
    let mut x_batch = Array::<f32, _>::zeros((batch_size, 28 * 28));
    let mut y_batch = Array::<u32, _>::zeros((batch_size,));

    let mut rng = ndarray_rand::rand::thread_rng();

    let gpu_state = unsafe { gpu_init() };
    let shaders = unsafe { load_shaders(&gpu_state) };
    let buffers = unsafe { create_buffers(&gpu_state, &shaders, batch_size as u64, &init_weights) };
    let cmd_bufs = unsafe {
        cmd_buffers(
            &gpu_state,
            &shaders,
            &buffers,
            batch_size as u32,
            step_size,
            reg,
        )
    };

    let initialized_time = initialized_time.elapsed();

    let mut rng_times = vec![];
    let mut batch_load_times = vec![];
    let mut submit_times = vec![];
    let mut execute_times = vec![];
    let mut per_iter_times = vec![];

    let mut next_batch = 0;

    let mut check: Option<Instant> = None;
    let mut per_iter = Instant::now();
    for i in 1..=num_iterations {
        // prepare the batches
        let then = Instant::now();
        r.as_slice_mut().unwrap().shuffle(&mut rng);
        for i in 0..batch_size {
            let index = r[i] as usize;
            x_batch.slice_mut(s![i, ..]).assign(&x.row(index));
            y_batch[i] = y[index];
        }
        rng_times.push(then.elapsed());

        let then = Instant::now();
        unsafe {
            load_batch_to_cpu(&gpu_state, &buffers, &x_batch, &y_batch);
            batch_load_times.push(then.elapsed());
            let info = vk::FenceCreateInfo::builder().build();
            let fence = gpu_state.device.create_fence(&info, ALLOCATOR).unwrap();
            let s = vk::SubmitInfo::builder()
                .command_buffers(&[cmd_bufs[next_batch]])
                // .signal_semaphores(signal_semaphores)
                .build();
            gpu_state
                .device
                .queue_submit(gpu_state.queue, &[s], fence)
                .unwrap();
            if let Some(check) = check {
                execute_times.push(check.elapsed());
            }
            if i == 1 {
                let s = [fence];
                gpu_state.device.wait_for_fences(&s, true, !0).unwrap();
            } else {
                let s = [fence, buffers.gpu_batches[next_batch ^ 1].fence];
                gpu_state.device.wait_for_fences(&s, true, !0).unwrap();
                let s = [buffers.gpu_batches[next_batch ^ 1].fence];
                gpu_state.device.reset_fences(&s).unwrap();
                per_iter_times.push(per_iter.elapsed());
                per_iter = Instant::now();
            }
            gpu_state.device.destroy_fence(fence, ALLOCATOR);
            let then = Instant::now();
            let j = [cmd_bufs[next_batch + 2]];
            let s = [vk::SubmitInfo::builder().command_buffers(&j).build()];
            gpu_state
                .device
                .queue_submit(gpu_state.queue, &s, buffers.gpu_batches[next_batch].fence)
                .unwrap();
            submit_times.push(then.elapsed());
            check = Some(Instant::now());
        }

        next_batch = next_batch ^ 1;
    }

    let finalize_time = Instant::now();

    let weights: Array2<f32>;
    let biases: Array2<f32>;
    let mut losses = vec![];

    unsafe {
        let info = vk::BufferCreateInfo::builder()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&[gpu_state.qf_index])
            .size(784 * 10 * 4)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .build();
        let copy_s_buffer = gpu_state.device.create_buffer(&info, ALLOCATOR).unwrap();
        let mem_reqs = gpu_state
            .device
            .get_buffer_memory_requirements(copy_s_buffer);
        let index = gpu_state
            .memorytype_index(
                &mem_reqs,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .unwrap();
        let info = vk::MemoryAllocateInfo::builder()
            .allocation_size(784 * 10 * 4)
            .memory_type_index(index)
            .build();
        let copy_mem = gpu_state.device.allocate_memory(&info, ALLOCATOR).unwrap();
        gpu_state
            .device
            .bind_buffer_memory(copy_s_buffer, copy_mem, 0)
            .unwrap();
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(3)
            .command_pool(gpu_state.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .build();
        let mut cbs = gpu_state
            .device
            .allocate_command_buffers(&info)
            .unwrap()
            .into_iter();
        let cb = cbs.next().unwrap();
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        gpu_state.device.begin_command_buffer(cb, &info).unwrap();
        gpu_state.device.cmd_copy_buffer(
            cb,
            buffers.w_buffer,
            copy_s_buffer,
            &[vk::BufferCopy::builder()
                .size(784 * 10 * 4)
                .src_offset(0)
                .dst_offset(0)
                .build()],
        );
        gpu_state.device.end_command_buffer(cb).unwrap();
        gpu_state
            .device
            .queue_submit(
                gpu_state.queue,
                &[vk::SubmitInfo::builder().command_buffers(&[cb]).build()],
                vk::Fence::null(),
            )
            .unwrap();
        gpu_state.device.queue_wait_idle(gpu_state.queue).unwrap(); // i should wait on a fence but there's nothing else running
        let ptr = gpu_state
            .device
            .map_memory(copy_mem, 0, 784 * 10 * 4, vk::MemoryMapFlags::empty())
            .unwrap();
        let s = std::slice::from_raw_parts(ptr as *const f32, 784 * 10);
        weights = Array2::from_shape_vec((784, 10), s.to_vec()).unwrap();
        // println!("{:?}", &s[..50]);
        gpu_state.device.unmap_memory(copy_mem);
        gpu_state
            .device
            .free_command_buffers(gpu_state.command_pool, &[cb]);

        let cb = cbs.next().unwrap();
        gpu_state.device.begin_command_buffer(cb, &info).unwrap();
        gpu_state.device.cmd_copy_buffer(
            cb,
            buffers.b_buffer,
            copy_s_buffer,
            &[vk::BufferCopy::builder()
                .size(10 * 4)
                .src_offset(0)
                .dst_offset(0)
                .build()],
        );
        gpu_state.device.end_command_buffer(cb).unwrap();
        gpu_state
            .device
            .queue_submit(
                gpu_state.queue,
                &[vk::SubmitInfo::builder().command_buffers(&[cb]).build()],
                vk::Fence::null(),
            )
            .unwrap();
        gpu_state.device.queue_wait_idle(gpu_state.queue).unwrap();
        let ptr = gpu_state
            .device
            .map_memory(copy_mem, 0, 10 * 4, vk::MemoryMapFlags::empty())
            .unwrap();
        let s = std::slice::from_raw_parts(ptr as *const f32, 10);
        // println!("{:?}", &s);
        biases = Array2::from_shape_vec((1, 10), s.to_vec()).unwrap();
        gpu_state.device.unmap_memory(copy_mem);
        gpu_state
            .device
            .free_command_buffers(gpu_state.command_pool, &[cb]);

        let cb = cbs.next().unwrap();
        gpu_state.device.begin_command_buffer(cb, &info).unwrap();
        gpu_state.device.cmd_copy_buffer(
            cb,
            buffers.loss_buffer,
            copy_s_buffer,
            &[vk::BufferCopy::builder()
                .size(1000 * 4)
                .src_offset(0)
                .dst_offset(0)
                .build()],
        );
        gpu_state.device.end_command_buffer(cb).unwrap();
        gpu_state
            .device
            .queue_submit(
                gpu_state.queue,
                &[vk::SubmitInfo::builder().command_buffers(&[cb]).build()],
                vk::Fence::null(),
            )
            .unwrap();
        gpu_state.device.queue_wait_idle(gpu_state.queue).unwrap();
        let ptr = gpu_state
            .device
            .map_memory(copy_mem, 0, 1000 * 4, vk::MemoryMapFlags::empty())
            .unwrap();
        let s = std::slice::from_raw_parts(ptr as *const f32, num_iterations as usize);
        // println!("{:?}", &s[..50]);
        losses.extend_from_slice(s);

        gpu_state.device.destroy_buffer(copy_s_buffer, ALLOCATOR);
        gpu_state.device.free_memory(copy_mem, ALLOCATOR);
    }

    unsafe {
        gpu_state.device.device_wait_idle().unwrap();
        gpu_state
            .device
            .free_command_buffers(gpu_state.command_pool, &cmd_bufs);
        destroy_buffer_state(&gpu_state, buffers);
        destroy_shaders(&gpu_state, shaders);
        destroy_gpu_state(gpu_state)
    };

    rng_times.sort();
    batch_load_times.sort();
    submit_times.sort();
    execute_times.sort();
    per_iter_times.sort();

    println!("Initialization time: {:?}", initialized_time);
    println!(
        "Rng and batch assign times: worst={:?}, median={:?}, average={:?}, best={:?}",
        rng_times.last().unwrap(),
        rng_times[rng_times.len() / 2],
        rng_times.iter().sum::<Duration>() / rng_times.len() as u32,
        rng_times.first().unwrap(),
    );
    println!(
        "Batch load times: worst={:?}, median={:?}, average={:?}, best={:?}",
        batch_load_times.last().unwrap(),
        batch_load_times[batch_load_times.len() / 2],
        batch_load_times.iter().sum::<Duration>() / batch_load_times.len() as u32,
        batch_load_times.first().unwrap(),
    );
    println!(
        "Submit times: worst={:?}, median={:?}, average={:?}, best={:?}",
        submit_times.last().unwrap(),
        submit_times[submit_times.len() / 2],
        submit_times.iter().sum::<Duration>() / submit_times.len() as u32,
        submit_times.first().unwrap(),
    );
    println!(
        "Execute times: worst={:?}, median={:?}, average={:?}, best={:?}",
        execute_times.last().unwrap(),
        execute_times[execute_times.len() / 2],
        execute_times.iter().sum::<Duration>() / execute_times.len() as u32,
        execute_times.first().unwrap(),
    );
    println!(
        "Per-iter times: worst={:?}, median={:?}, average={:?}, best={:?}",
        per_iter_times.last().unwrap(),
        per_iter_times[per_iter_times.len() / 2],
        per_iter_times.iter().sum::<Duration>() / per_iter_times.len() as u32,
        per_iter_times.first().unwrap(),
    );
    println!("Finalize time: {:?}", finalize_time.elapsed());

    (weights, biases, losses)
}

unsafe fn cmd_buffers(
    state: &GPUState,
    shaders: &ShaderBundle,
    buffers: &BufferState,
    batch_size: u32,
    step_size: f32,
    reg: f32,
) -> Vec<vk::CommandBuffer> {
    let info = vk::CommandBufferAllocateInfo::builder()
        .command_buffer_count(4)
        .command_pool(state.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .build();
    let cbs = state.device.allocate_command_buffers(&info).unwrap();
    let info = vk::CommandBufferBeginInfo::builder();
    state.device.begin_command_buffer(cbs[0], &info).unwrap();
    state.device.cmd_copy_buffer(
        cbs[0],
        buffers.xbatch_cpu_buffer,
        buffers.gpu_batches[0].xbatch_gpu_buffer,
        &[vk::BufferCopy::builder()
            .size(buffers.xbatch_size)
            .src_offset(0)
            .dst_offset(0)
            .build()],
    );
    state.device.cmd_copy_buffer(
        cbs[0],
        buffers.ybatch_cpu_buffer,
        buffers.gpu_batches[0].ybatch_gpu_buffer,
        &[vk::BufferCopy::builder()
            .size(buffers.ybatch_size)
            .src_offset(0)
            .dst_offset(0)
            .build()],
    );
    state.device.end_command_buffer(cbs[0]).unwrap();
    state.device.begin_command_buffer(cbs[1], &info).unwrap();
    state.device.cmd_copy_buffer(
        cbs[1],
        buffers.xbatch_cpu_buffer,
        buffers.gpu_batches[1].xbatch_gpu_buffer,
        &[vk::BufferCopy::builder()
            .size(buffers.xbatch_size)
            .src_offset(0)
            .dst_offset(0)
            .build()],
    );
    state.device.cmd_copy_buffer(
        cbs[1],
        buffers.ybatch_cpu_buffer,
        buffers.gpu_batches[1].ybatch_gpu_buffer,
        &[vk::BufferCopy::builder()
            .size(buffers.ybatch_size)
            .src_offset(0)
            .dst_offset(0)
            .build()],
    );
    state.device.end_command_buffer(cbs[1]).unwrap();
    state.device.begin_command_buffer(cbs[2], &info).unwrap();
    state.device.cmd_bind_pipeline(
        cbs[2],
        vk::PipelineBindPoint::COMPUTE,
        shaders.scores_pipeline,
    );
    state.device.cmd_bind_descriptor_sets(
        cbs[2],
        vk::PipelineBindPoint::COMPUTE,
        shaders.scores_pipeline_layout,
        0,
        &[
            buffers.gpu_batches[0].batch_ds,
            buffers.weights_ds,
            buffers.scores_ds,
        ],
        &[],
    );
    state.device.cmd_dispatch(cbs[2], 1, batch_size, 10);
    state.device.cmd_pipeline_barrier(
        cbs[2],
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::BY_REGION,
        &[vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .build()],
        &[],
        &[],
    );
    state.device.cmd_bind_pipeline(
        cbs[2],
        vk::PipelineBindPoint::COMPUTE,
        shaders.scores_2_pipeline,
    );
    state.device.cmd_bind_descriptor_sets(
        cbs[2],
        vk::PipelineBindPoint::COMPUTE,
        shaders.scores_2_pipeline_layout,
        0,
        &[buffers.scores_ds],
        &[],
    );
    state.device.cmd_dispatch(cbs[2], batch_size / 64, 1, 1);
    state.device.cmd_pipeline_barrier(
        cbs[2],
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::BY_REGION,
        &[vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .build()],
        &[],
        &[],
    );
    state.device.cmd_bind_pipeline(
        cbs[2],
        vk::PipelineBindPoint::COMPUTE,
        shaders.loss_pipeline,
    );
    state.device.cmd_bind_descriptor_sets(
        cbs[2],
        vk::PipelineBindPoint::COMPUTE,
        shaders.loss_pipeline_layout,
        0,
        &[
            buffers.gpu_batches[0].batch_ds,
            buffers.scores_ds,
            buffers.loss_ds,
        ],
        &[],
    );
    state.device.cmd_dispatch(cbs[2], 1, 1, 1);
    state.device.cmd_pipeline_barrier(
        cbs[2],
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::BY_REGION,
        &[vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .build()],
        &[],
        &[],
    );
    state.device.cmd_bind_descriptor_sets(
        cbs[2],
        vk::PipelineBindPoint::COMPUTE,
        shaders.gradient_pipeline_layout,
        0,
        &[buffers.gpu_batches[0].batch_ds, buffers.scores_ds],
        &[],
    );
    state.device.cmd_bind_pipeline(
        cbs[2],
        vk::PipelineBindPoint::COMPUTE,
        shaders.gradient_pipeline,
    );
    state.device.cmd_dispatch(cbs[2], 1, 1, 1);
    state.device.cmd_pipeline_barrier(
        cbs[2],
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::BY_REGION,
        &[vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .build()],
        &[],
        &[],
    );
    state.device.cmd_bind_descriptor_sets(
        cbs[2],
        vk::PipelineBindPoint::COMPUTE,
        shaders.gradient_2_pipeline_layout,
        0,
        &[
            buffers.gpu_batches[0].batch_ds,
            buffers.weights_ds,
            buffers.scores_ds,
        ],
        &[],
    );
    let ps = [step_size, reg];
    state.device.cmd_push_constants(
        cbs[2],
        shaders.gradient_2_pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        std::slice::from_raw_parts(ps.as_ptr() as *const u8, 8),
    );
    state.device.cmd_bind_pipeline(
        cbs[2],
        vk::PipelineBindPoint::COMPUTE,
        shaders.gradient_2_pipeline,
    );
    state.device.cmd_dispatch(cbs[2], 1, 784, 10);
    state.device.end_command_buffer(cbs[2]).unwrap();

    state.device.begin_command_buffer(cbs[3], &info).unwrap();
    state.device.cmd_bind_pipeline(
        cbs[3],
        vk::PipelineBindPoint::COMPUTE,
        shaders.scores_pipeline,
    );
    state.device.cmd_bind_descriptor_sets(
        cbs[3],
        vk::PipelineBindPoint::COMPUTE,
        shaders.scores_pipeline_layout,
        0,
        &[
            buffers.gpu_batches[1].batch_ds,
            buffers.weights_ds,
            buffers.scores_ds,
        ],
        &[],
    );
    state.device.cmd_dispatch(cbs[3], 1, batch_size, 10);
    state.device.cmd_pipeline_barrier(
        cbs[3],
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::BY_REGION,
        &[vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .build()],
        &[],
        &[],
    );
    state.device.cmd_bind_pipeline(
        cbs[3],
        vk::PipelineBindPoint::COMPUTE,
        shaders.scores_2_pipeline,
    );
    state.device.cmd_bind_descriptor_sets(
        cbs[3],
        vk::PipelineBindPoint::COMPUTE,
        shaders.scores_2_pipeline_layout,
        0,
        &[buffers.scores_ds],
        &[],
    );
    state.device.cmd_dispatch(cbs[3], batch_size / 64, 1, 1);
    state.device.cmd_pipeline_barrier(
        cbs[3],
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::BY_REGION,
        &[vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .build()],
        &[],
        &[],
    );
    state.device.cmd_bind_pipeline(
        cbs[3],
        vk::PipelineBindPoint::COMPUTE,
        shaders.loss_pipeline,
    );
    state.device.cmd_bind_descriptor_sets(
        cbs[3],
        vk::PipelineBindPoint::COMPUTE,
        shaders.loss_pipeline_layout,
        0,
        &[
            buffers.gpu_batches[1].batch_ds,
            buffers.scores_ds,
            buffers.loss_ds,
        ],
        &[],
    );
    state.device.cmd_dispatch(cbs[3], 1, 1, 1);
    state.device.cmd_pipeline_barrier(
        cbs[3],
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::BY_REGION,
        &[vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .build()],
        &[],
        &[],
    );
    state.device.cmd_bind_descriptor_sets(
        cbs[3],
        vk::PipelineBindPoint::COMPUTE,
        shaders.gradient_pipeline_layout,
        0,
        &[buffers.gpu_batches[1].batch_ds, buffers.scores_ds],
        &[],
    );
    state.device.cmd_bind_pipeline(
        cbs[3],
        vk::PipelineBindPoint::COMPUTE,
        shaders.gradient_pipeline,
    );
    state.device.cmd_dispatch(cbs[3], 1, 1, 1);
    state.device.cmd_pipeline_barrier(
        cbs[3],
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::BY_REGION,
        &[vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .build()],
        &[],
        &[],
    );
    state.device.cmd_bind_descriptor_sets(
        cbs[3],
        vk::PipelineBindPoint::COMPUTE,
        shaders.gradient_2_pipeline_layout,
        0,
        &[
            buffers.gpu_batches[1].batch_ds,
            buffers.weights_ds,
            buffers.scores_ds,
        ],
        &[],
    );
    let ps = [step_size, reg];
    state.device.cmd_push_constants(
        cbs[3],
        shaders.gradient_2_pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        std::slice::from_raw_parts(ps.as_ptr() as *const u8, 8),
    );
    state.device.cmd_bind_pipeline(
        cbs[3],
        vk::PipelineBindPoint::COMPUTE,
        shaders.gradient_2_pipeline,
    );
    state.device.cmd_dispatch(cbs[3], 1, 784, 10);
    state.device.end_command_buffer(cbs[3]).unwrap();
    cbs
}

struct GPUState {
    #[allow(unused)]
    // we save the entry so it isn't dropped
    entry: Entry,
    instance: Instance,
    debug_utils_loader: DebugUtils,
    debug_callback: DebugUtilsMessengerEXT,
    device: Device,
    queue: vk::Queue,
    qf_index: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    command_pool: vk::CommandPool,
}

impl GPUState {
    fn memorytype_index(
        &self,
        reqs: &vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        self.memory_properties.memory_types[..self.memory_properties.memory_type_count as _]
            .iter()
            .enumerate()
            .find(|(index, memory_type)| {
                (1 << index) & reqs.memory_type_bits != 0
                    && memory_type.property_flags & flags == flags
            })
            .map(|(index, _)| index as _)
    }
}

unsafe fn load_batch_to_cpu(
    state: &GPUState,
    buffers: &BufferState,
    x_batch: &Array2<f32>,
    y_batch: &Array1<u32>,
) {
    let ptr = state
        .device
        .map_memory(
            buffers.xbatch_cpu_memory,
            0,
            buffers.xbatch_size,
            vk::MemoryMapFlags::empty(),
        )
        .unwrap();
    std::ptr::copy(
        x_batch.as_slice().unwrap().as_ptr() as *const u8,
        ptr as *mut u8,
        buffers.xbatch_size as usize,
    );
    state.device.unmap_memory(buffers.xbatch_cpu_memory);
    let ptr = state
        .device
        .map_memory(
            buffers.ybatch_cpu_memory,
            0,
            buffers.ybatch_size,
            vk::MemoryMapFlags::empty(),
        )
        .unwrap();
    std::ptr::copy(
        y_batch.as_slice().unwrap().as_ptr() as *const u8,
        ptr as *mut u8,
        buffers.ybatch_size as usize,
    );
    state.device.unmap_memory(buffers.ybatch_cpu_memory);
}

struct BufferState {
    descriptor_pool: vk::DescriptorPool,
    xbatch_cpu_buffer: vk::Buffer,
    xbatch_cpu_memory: vk::DeviceMemory,
    xbatch_size: u64,
    ybatch_cpu_buffer: vk::Buffer,
    ybatch_cpu_memory: vk::DeviceMemory,
    ybatch_size: u64,
    gpu_batches: [GPUBatch; 2],
    w_buffer: vk::Buffer,
    w_memory: vk::DeviceMemory,
    b_buffer: vk::Buffer,
    b_memory: vk::DeviceMemory,
    weights_ds: vk::DescriptorSet,
    scores_buffer: vk::Buffer,
    scores_memory: vk::DeviceMemory,
    scores_ds: vk::DescriptorSet,
    loss_buffer: vk::Buffer,
    loss_memory: vk::DeviceMemory,
    loss_ds: vk::DescriptorSet,
}

struct GPUBatch {
    xbatch_gpu_buffer: vk::Buffer,
    xbatch_gpu_memory: vk::DeviceMemory,
    ybatch_gpu_buffer: vk::Buffer,
    ybatch_gpu_memory: vk::DeviceMemory,
    batch_ds: vk::DescriptorSet,
    fence: vk::Fence,
}

unsafe fn destroy_buffer_state(state: &GPUState, buffers: BufferState) {
    state
        .device
        .destroy_descriptor_pool(buffers.descriptor_pool, ALLOCATOR);
    for batch in buffers.gpu_batches.iter() {
        state.device.destroy_fence(batch.fence, ALLOCATOR);
        state
            .device
            .destroy_buffer(batch.xbatch_gpu_buffer, ALLOCATOR);
        state.device.free_memory(batch.xbatch_gpu_memory, ALLOCATOR);
        state
            .device
            .destroy_buffer(batch.ybatch_gpu_buffer, ALLOCATOR);
        state.device.free_memory(batch.ybatch_gpu_memory, ALLOCATOR);
    }
    state
        .device
        .destroy_buffer(buffers.scores_buffer, ALLOCATOR);
    state.device.free_memory(buffers.scores_memory, ALLOCATOR);
    state.device.destroy_buffer(buffers.w_buffer, ALLOCATOR);
    state.device.free_memory(buffers.w_memory, ALLOCATOR);
    state.device.destroy_buffer(buffers.b_buffer, ALLOCATOR);
    state.device.free_memory(buffers.b_memory, ALLOCATOR);
    state.device.destroy_buffer(buffers.loss_buffer, ALLOCATOR);
    state.device.free_memory(buffers.loss_memory, ALLOCATOR);
    state
        .device
        .destroy_buffer(buffers.xbatch_cpu_buffer, ALLOCATOR);
    state
        .device
        .free_memory(buffers.xbatch_cpu_memory, ALLOCATOR);
    state
        .device
        .destroy_buffer(buffers.ybatch_cpu_buffer, ALLOCATOR);
    state
        .device
        .free_memory(buffers.ybatch_cpu_memory, ALLOCATOR);
}

unsafe fn create_buffers(
    state: &GPUState,
    shaders: &ShaderBundle,
    batch_size: u64,
    init_weights: &Array2<f32>,
) -> BufferState {
    let i = [vk::DescriptorPoolSize::builder()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(16)
        .build()];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .max_sets(8)
        .pool_sizes(&i)
        .build();
    let descriptor_pool = state
        .device
        .create_descriptor_pool(&info, ALLOCATOR)
        .unwrap();

    let xbatch_size = size_of::<f32>() as u64 * 28 * 28 * batch_size;
    let q = [state.qf_index];
    let info = vk::BufferCreateInfo::builder()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(&q)
        .size(xbatch_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .build();
    let xbatch_cpu_buffer = state.device.create_buffer(&info, ALLOCATOR).unwrap();
    let mem_reqs = state
        .device
        .get_buffer_memory_requirements(xbatch_cpu_buffer);
    let index = state
        .memorytype_index(
            &mem_reqs,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(xbatch_size)
        .memory_type_index(index)
        .build();
    let xbatch_cpu_memory = state.device.allocate_memory(&info, ALLOCATOR).unwrap();
    state
        .device
        .bind_buffer_memory(xbatch_cpu_buffer, xbatch_cpu_memory, 0)
        .unwrap();

    let ybatch_size = size_of::<f32>() as u64 * batch_size;
    let q = [state.qf_index];
    let info = vk::BufferCreateInfo::builder()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(&q)
        .size(ybatch_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .build();
    let ybatch_cpu_buffer = state.device.create_buffer(&info, ALLOCATOR).unwrap();
    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(ybatch_size)
        .memory_type_index(index)
        .build();
    let ybatch_cpu_memory = state.device.allocate_memory(&info, ALLOCATOR).unwrap();
    state
        .device
        .bind_buffer_memory(ybatch_cpu_buffer, ybatch_cpu_memory, 0)
        .unwrap();

    let c_b = || {
        let q = [state.qf_index];
        let info = vk::BufferCreateInfo::builder()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&q)
            .size(xbatch_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER)
            .build();
        let xbatch_gpu_buffer = state.device.create_buffer(&info, ALLOCATOR).unwrap();
        let mem_reqs = state
            .device
            .get_buffer_memory_requirements(xbatch_gpu_buffer);
        let index = state
            .memorytype_index(&mem_reqs, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .unwrap();
        let info = vk::MemoryAllocateInfo::builder()
            .allocation_size(xbatch_size)
            .memory_type_index(index)
            .build();
        let xbatch_gpu_memory = state.device.allocate_memory(&info, ALLOCATOR).unwrap();
        state
            .device
            .bind_buffer_memory(xbatch_gpu_buffer, xbatch_gpu_memory, 0)
            .unwrap();
        let ybatch_size = size_of::<f32>() as u64 * batch_size;
        let info = vk::BufferCreateInfo::builder()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&q)
            .size(ybatch_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER)
            .build();
        let ybatch_gpu_buffer = state.device.create_buffer(&info, ALLOCATOR).unwrap();
        let info = vk::MemoryAllocateInfo::builder()
            .allocation_size(ybatch_size)
            .memory_type_index(index)
            .build();
        let ybatch_gpu_memory = state.device.allocate_memory(&info, ALLOCATOR).unwrap();
        state
            .device
            .bind_buffer_memory(ybatch_gpu_buffer, ybatch_gpu_memory, 0)
            .unwrap();
        let l = [shaders.batch_ds_layout];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&l);
        let batch_ds = state
            .device
            .allocate_descriptor_sets(&info)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        state.device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::builder()
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .dst_array_element(0)
                    .dst_binding(0)
                    .dst_set(batch_ds)
                    .buffer_info(&[vk::DescriptorBufferInfo::builder()
                        .buffer(xbatch_gpu_buffer)
                        .offset(0)
                        .range(xbatch_size)
                        .build()])
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .dst_array_element(0)
                    .dst_binding(1)
                    .dst_set(batch_ds)
                    .buffer_info(&[vk::DescriptorBufferInfo::builder()
                        .buffer(ybatch_gpu_buffer)
                        .offset(0)
                        .range(ybatch_size)
                        .build()])
                    .build(),
            ],
            &[],
        );
        GPUBatch {
            xbatch_gpu_buffer,
            xbatch_gpu_memory,
            ybatch_gpu_buffer,
            ybatch_gpu_memory,
            batch_ds,
            fence: state
                .device
                .create_fence(&vk::FenceCreateInfo::builder().build(), ALLOCATOR)
                .unwrap(),
        }
    };
    let gpu_batches = [c_b(), c_b()];

    let q = [state.qf_index];
    let w_size = size_of::<f32>() as u64 * 28 * 28 * 10;
    let info = vk::BufferCreateInfo::builder()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(&q)
        .size(w_size)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .build();
    let w_buffer = state.device.create_buffer(&info, ALLOCATOR).unwrap();
    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(w_size)
        .memory_type_index(index)
        .build();
    let w_memory = state.device.allocate_memory(&info, ALLOCATOR).unwrap();
    state
        .device
        .bind_buffer_memory(w_buffer, w_memory, 0)
        .unwrap();

    let info = vk::BufferCreateInfo::builder()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(&q)
        .size(4 * 10000) // just in case
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC)
        .build();
    let loss_buffer = state.device.create_buffer(&info, ALLOCATOR).unwrap();
    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(4 * 10000)
        .memory_type_index(index)
        .build();
    let loss_memory = state.device.allocate_memory(&info, ALLOCATOR).unwrap();
    state
        .device
        .bind_buffer_memory(loss_buffer, loss_memory, 0)
        .unwrap();

    let l = [shaders.loss_ds_layout];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .set_layouts(&l)
        .descriptor_pool(descriptor_pool)
        .build();
    let loss_ds = state
        .device
        .allocate_descriptor_sets(&info)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    state.device.update_descriptor_sets(
        &[vk::WriteDescriptorSet::builder()
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .dst_array_element(0)
            .dst_binding(0)
            .dst_set(loss_ds)
            .buffer_info(&[vk::DescriptorBufferInfo::builder()
                .buffer(loss_buffer)
                .offset(0)
                .range(4 * 10000) // just to be safe
                .build()])
            .build()],
        &[],
    );

    {
        let info = vk::BufferCreateInfo::builder()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&q)
            .size(w_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .build();
        let copy_w_buffer = state.device.create_buffer(&info, ALLOCATOR).unwrap();
        let mem_reqs = state.device.get_buffer_memory_requirements(copy_w_buffer);
        let index = state
            .memorytype_index(
                &mem_reqs,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .unwrap();
        let info = vk::MemoryAllocateInfo::builder()
            .allocation_size(xbatch_size)
            .memory_type_index(index)
            .build();
        let copy_mem = state.device.allocate_memory(&info, ALLOCATOR).unwrap();
        state
            .device
            .bind_buffer_memory(copy_w_buffer, copy_mem, 0)
            .unwrap();
        let ptr = state
            .device
            .map_memory(copy_mem, 0, w_size, vk::MemoryMapFlags::empty())
            .unwrap();
        std::ptr::copy(
            init_weights.as_slice().unwrap().as_ptr() as *const u8,
            ptr as *mut u8,
            w_size as usize,
        );
        state.device.unmap_memory(copy_mem);
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(state.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .build();
        let cb = state
            .device
            .allocate_command_buffers(&info)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        state.device.begin_command_buffer(cb, &info).unwrap();
        let b = vk::BufferCopy::builder()
            .size(w_size)
            .src_offset(0)
            .dst_offset(0)
            .build();
        state
            .device
            .cmd_copy_buffer(cb, copy_w_buffer, w_buffer, &[b]);
        state.device.end_command_buffer(cb).unwrap();
        let v = [vk::SubmitInfo::builder().command_buffers(&[cb]).build()];
        state
            .device
            .queue_submit(state.queue, &v, vk::Fence::null())
            .unwrap();
        let c = [cb];
        state.device.queue_wait_idle(state.queue).unwrap(); // i should wait on a fence but there's nothing else running
        state.device.free_command_buffers(state.command_pool, &c);
        state.device.destroy_buffer(copy_w_buffer, ALLOCATOR);
        state.device.free_memory(copy_mem, ALLOCATOR);
    }

    let b_size = size_of::<f32>() as u64 * 10;
    let info = vk::BufferCreateInfo::builder()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(&q)
        .size(b_size)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC)
        .build();
    let b_buffer = state.device.create_buffer(&info, ALLOCATOR).unwrap();
    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(b_size.max(mem_reqs.size))
        .memory_type_index(index)
        .build();
    let b_memory = state.device.allocate_memory(&info, ALLOCATOR).unwrap();
    state
        .device
        .bind_buffer_memory(b_buffer, b_memory, 0)
        .unwrap();

    let l = [shaders.weights_ds_layout];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&l);
    let weights_ds = state
        .device
        .allocate_descriptor_sets(&info)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    state.device.update_descriptor_sets(
        &[
            vk::WriteDescriptorSet::builder()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .dst_array_element(0)
                .dst_binding(0)
                .dst_set(weights_ds)
                .buffer_info(&[vk::DescriptorBufferInfo::builder()
                    .buffer(w_buffer)
                    .offset(0)
                    .range(w_size)
                    .build()])
                .build(),
            vk::WriteDescriptorSet::builder()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .dst_array_element(0)
                .dst_binding(1)
                .dst_set(weights_ds)
                .buffer_info(&[vk::DescriptorBufferInfo::builder()
                    .buffer(b_buffer)
                    .offset(0)
                    .range(b_size)
                    .build()])
                .build(),
        ],
        &[],
    );

    let scores_size = size_of::<f32>() as u64 * batch_size * 10;
    let info = vk::BufferCreateInfo::builder()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(&q)
        .size(scores_size)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC)
        .build();
    let scores_buffer = state.device.create_buffer(&info, ALLOCATOR).unwrap();
    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(scores_size)
        .memory_type_index(index)
        .build();
    let scores_memory = state.device.allocate_memory(&info, ALLOCATOR).unwrap();
    state
        .device
        .bind_buffer_memory(scores_buffer, scores_memory, 0)
        .unwrap();

    let l = [shaders.scores_ds_layout];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&l);
    let scores_ds = state
        .device
        .allocate_descriptor_sets(&info)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let b = [vk::DescriptorBufferInfo::builder()
        .buffer(scores_buffer)
        .offset(0)
        .range(scores_size)
        .build()];
    state.device.update_descriptor_sets(
        &[vk::WriteDescriptorSet::builder()
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .dst_array_element(0)
            .dst_binding(0)
            .dst_set(scores_ds)
            .buffer_info(&b)
            .build()],
        &[],
    );

    BufferState {
        descriptor_pool,

        xbatch_cpu_buffer,
        xbatch_cpu_memory,
        xbatch_size,

        ybatch_cpu_buffer,
        ybatch_cpu_memory,
        ybatch_size,

        w_buffer,
        w_memory,
        b_buffer,
        b_memory,
        scores_buffer,
        scores_memory,
        loss_buffer,
        loss_memory,

        weights_ds,
        scores_ds,
        loss_ds,

        gpu_batches,
    }
}

struct ShaderBundle {
    scores: vk::ShaderModule,
    scores_2: vk::ShaderModule,
    loss: vk::ShaderModule,
    gradient: vk::ShaderModule,
    gradient_2: vk::ShaderModule,

    batch_ds_layout: vk::DescriptorSetLayout,
    weights_ds_layout: vk::DescriptorSetLayout,
    scores_ds_layout: vk::DescriptorSetLayout,
    loss_ds_layout: vk::DescriptorSetLayout,

    scores_pipeline_layout: vk::PipelineLayout,
    scores_pipeline: vk::Pipeline,

    scores_2_pipeline_layout: vk::PipelineLayout,
    scores_2_pipeline: vk::Pipeline,

    loss_pipeline_layout: vk::PipelineLayout,
    loss_pipeline: vk::Pipeline,

    gradient_pipeline_layout: vk::PipelineLayout,
    gradient_pipeline: vk::Pipeline,

    gradient_2_pipeline_layout: vk::PipelineLayout,
    gradient_2_pipeline: vk::Pipeline,
}

unsafe fn destroy_shaders(state: &GPUState, shaders: ShaderBundle) {
    state
        .device
        .destroy_descriptor_set_layout(shaders.batch_ds_layout, ALLOCATOR);
    state
        .device
        .destroy_descriptor_set_layout(shaders.weights_ds_layout, ALLOCATOR);
    state
        .device
        .destroy_descriptor_set_layout(shaders.scores_ds_layout, ALLOCATOR);
    state
        .device
        .destroy_descriptor_set_layout(shaders.loss_ds_layout, ALLOCATOR);
    state
        .device
        .destroy_pipeline_layout(shaders.scores_pipeline_layout, ALLOCATOR);
    state
        .device
        .destroy_pipeline_layout(shaders.scores_2_pipeline_layout, ALLOCATOR);
    state
        .device
        .destroy_pipeline_layout(shaders.loss_pipeline_layout, ALLOCATOR);
    state
        .device
        .destroy_pipeline_layout(shaders.gradient_pipeline_layout, ALLOCATOR);
    state
        .device
        .destroy_pipeline_layout(shaders.gradient_2_pipeline_layout, ALLOCATOR);
    state
        .device
        .destroy_pipeline(shaders.scores_pipeline, ALLOCATOR);
    state
        .device
        .destroy_pipeline(shaders.scores_2_pipeline, ALLOCATOR);
    state
        .device
        .destroy_pipeline(shaders.loss_pipeline, ALLOCATOR);
    state
        .device
        .destroy_pipeline(shaders.gradient_pipeline, ALLOCATOR);
    state
        .device
        .destroy_pipeline(shaders.gradient_2_pipeline, ALLOCATOR);
    state
        .device
        .destroy_shader_module(shaders.scores, ALLOCATOR);
    state
        .device
        .destroy_shader_module(shaders.scores_2, ALLOCATOR);
    state.device.destroy_shader_module(shaders.loss, ALLOCATOR);
    state
        .device
        .destroy_shader_module(shaders.gradient, ALLOCATOR);
    state
        .device
        .destroy_shader_module(shaders.gradient_2, ALLOCATOR);
}

unsafe fn load_shaders(state: &GPUState) -> ShaderBundle {
    let code_scores_shader =
        ash::util::read_spv(&mut Cursor::new(include_bytes!("scores.spv"))).unwrap();
    let code_scores_2_shader =
        ash::util::read_spv(&mut Cursor::new(include_bytes!("scores2.spv"))).unwrap();
    let code_loss_shader =
        ash::util::read_spv(&mut Cursor::new(include_bytes!("loss.spv"))).unwrap();
    let code_gradient_shader =
        ash::util::read_spv(&mut Cursor::new(include_bytes!("gradient.spv"))).unwrap();
    let code_gradient_2_shader =
        ash::util::read_spv(&mut Cursor::new(include_bytes!("gradient2.spv"))).unwrap();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code(&code_scores_shader)
        .build();
    let scores = state.device.create_shader_module(&info, ALLOCATOR).unwrap();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code(&code_scores_2_shader)
        .build();
    let scores_2 = state.device.create_shader_module(&info, ALLOCATOR).unwrap();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code(&code_loss_shader)
        .build();
    let loss = state.device.create_shader_module(&info, ALLOCATOR).unwrap();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code(&code_gradient_shader)
        .build();
    let gradient = state.device.create_shader_module(&info, ALLOCATOR).unwrap();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code(&code_gradient_2_shader)
        .build();
    let gradient_2 = state.device.create_shader_module(&info, ALLOCATOR).unwrap();

    let b = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(&b)
        .build();
    let batch_ds_layout = state
        .device
        .create_descriptor_set_layout(&info, ALLOCATOR)
        .unwrap();
    let b = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(&b)
        .build();
    let weights_ds_layout = state
        .device
        .create_descriptor_set_layout(&info, ALLOCATOR)
        .unwrap();

    let b = [vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .build()];
    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(&b)
        .build();
    let loss_ds_layout = state
        .device
        .create_descriptor_set_layout(&info, ALLOCATOR)
        .unwrap();

    let b = [vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .build()];
    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(&b)
        .build();
    let scores_ds_layout = state
        .device
        .create_descriptor_set_layout(&info, ALLOCATOR)
        .unwrap();

    let info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&[batch_ds_layout, weights_ds_layout, scores_ds_layout])
        .build();
    let scores_pipeline_layout = state
        .device
        .create_pipeline_layout(&info, ALLOCATOR)
        .unwrap();
    let b = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(scores)
        .name(CStr::from_bytes_with_nul(b"main\0").unwrap())
        .build();
    let scores_pipeline_info = vk::ComputePipelineCreateInfo::builder()
        .base_pipeline_index(0)
        .stage(b)
        .layout(scores_pipeline_layout)
        .build();
    let b = [scores_ds_layout];
    let info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&b)
        .build();
    let scores_2_pipeline_layout = state
        .device
        .create_pipeline_layout(&info, ALLOCATOR)
        .unwrap();
    let b = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(scores_2)
        .name(CStr::from_bytes_with_nul(b"main\0").unwrap())
        .build();
    let scores_2_pipeline_info = vk::ComputePipelineCreateInfo::builder()
        .base_pipeline_index(0)
        .stage(b)
        .layout(scores_2_pipeline_layout)
        .build();

    let b = [batch_ds_layout, scores_ds_layout, loss_ds_layout];
    let info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&b)
        // .push_constant_ranges(&[vk::PushConstantRange::builder()
        //     .size(4)
        //     .offset(0)
        //     .stage_flags(vk::ShaderStageFlags::COMPUTE)
        //     .build()])
        .build();
    let loss_pipeline_layout = state
        .device
        .create_pipeline_layout(&info, ALLOCATOR)
        .unwrap();
    let b = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(loss)
        .name(CStr::from_bytes_with_nul(b"main\0").unwrap())
        .build();
    let loss_pipeline_info = vk::ComputePipelineCreateInfo::builder()
        .base_pipeline_index(0)
        .stage(b)
        .layout(loss_pipeline_layout)
        .build();

    let b = [batch_ds_layout, scores_ds_layout];
    let info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&b)
        .build();
    let gradient_pipeline_layout = state
        .device
        .create_pipeline_layout(&info, ALLOCATOR)
        .unwrap();
    let b = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(gradient)
        .name(CStr::from_bytes_with_nul(b"main\0").unwrap())
        .build();
    let gradient_pipeline_info = vk::ComputePipelineCreateInfo::builder()
        .base_pipeline_index(0)
        .stage(b)
        .layout(gradient_pipeline_layout)
        .build();
    let b = [vk::PushConstantRange::builder()
        .size(8)
        .offset(0)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .build()];
    let a = [batch_ds_layout, weights_ds_layout, scores_ds_layout];
    let info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&a)
        .push_constant_ranges(&b)
        .build();
    let gradient_2_pipeline_layout = state
        .device
        .create_pipeline_layout(&info, ALLOCATOR)
        .unwrap();
    let s = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(gradient_2)
        .name(CStr::from_bytes_with_nul(b"main\0").unwrap())
        .build();
    let gradient_2_pipeline_info = vk::ComputePipelineCreateInfo::builder()
        .base_pipeline_index(0)
        .stage(s)
        .layout(gradient_2_pipeline_layout)
        .build();
    let s = [
        scores_pipeline_info,
        scores_2_pipeline_info,
        loss_pipeline_info,
        gradient_pipeline_info,
        gradient_2_pipeline_info,
    ];
    let mut pipelines = state
        .device
        .create_compute_pipelines(vk::PipelineCache::null(), &s, ALLOCATOR)
        .unwrap()
        .into_iter();
    let scores_pipeline = pipelines.next().unwrap();
    let scores_2_pipeline = pipelines.next().unwrap();
    let loss_pipeline = pipelines.next().unwrap();
    let gradient_pipeline = pipelines.next().unwrap();
    let gradient_2_pipeline = pipelines.next().unwrap();

    ShaderBundle {
        scores,
        scores_2,
        loss,
        gradient,
        gradient_2,
        batch_ds_layout,
        weights_ds_layout,
        scores_ds_layout,
        loss_ds_layout,
        scores_pipeline_layout,
        scores_pipeline,
        scores_2_pipeline_layout,
        scores_2_pipeline,
        loss_pipeline_layout,
        loss_pipeline,
        gradient_pipeline_layout,
        gradient_pipeline,
        gradient_2_pipeline_layout,
        gradient_2_pipeline,
    }
}

unsafe fn destroy_gpu_state(state: GPUState) {
    state
        .device
        .destroy_command_pool(state.command_pool, ALLOCATOR);
    state.device.destroy_device(ALLOCATOR);
    state
        .debug_utils_loader
        .destroy_debug_utils_messenger(state.debug_callback, ALLOCATOR);
    state.instance.destroy_instance(ALLOCATOR);
}

unsafe fn gpu_init() -> GPUState {
    let entry = Entry::load().expect("Could not link the Vulkan libraries");

    let app_name = CStr::from_bytes_with_nul(b"FCNet GPU Implementation\0").unwrap();
    // minimum api version is Vulkan 1.1, for float atomics support
    let appinfo = vk::ApplicationInfo::builder()
        .application_name(app_name)
        .application_version(0)
        .engine_name(app_name)
        .engine_version(0)
        .api_version(vk::make_api_version(0, 1, 1, 0));

    let layer_names = [CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0")
        .unwrap()
        .as_ptr()];
    let extension_names = [DebugUtils::name().as_ptr()];
    let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&appinfo)
        .enabled_layer_names(&layer_names)
        .enabled_extension_names(&extension_names);
    let instance: Instance = entry.create_instance(&create_info, None).unwrap();

    let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));
    let debug_utils_loader = DebugUtils::new(&entry, &instance);
    let debug_callback = debug_utils_loader
        .create_debug_utils_messenger(&debug_info, None)
        .unwrap();

    let physical_devices = instance.enumerate_physical_devices().unwrap();

    let physical_device = physical_devices
        .iter()
        .find_map(|&p_d| {
            let properties = instance.get_physical_device_properties(p_d);
            println!(
                "Considering {:?}\nDevice type: {:?}",
                CStr::from_ptr(properties.device_name.as_ptr()),
                properties.device_type
            );

            let mut float_atomic_features =
                vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
            let mut f2s =
                vk::PhysicalDeviceFeatures2::builder().push_next(&mut float_atomic_features);
            instance.get_physical_device_features2(p_d, &mut f2s);
            // println!("{:?}", float_atomic_features);
            if !(float_atomic_features.shader_shared_float32_atomics == 1
                && float_atomic_features.shader_shared_float32_atomic_add == 1)
            {
                println!("Device does not support the required float atomic features!");
                return None;
            }

            let extension_properties = instance.enumerate_device_extension_properties(p_d).unwrap();
            extension_properties.iter().find(|p| {
                CStr::from_ptr(p.extension_name.as_ptr()) == vk::ExtShaderAtomicFloatFn::name()
            })?;
            if properties.limits.max_compute_work_group_size[0] < 28 * 28 {
                println!(
                    "Device's max compute workgroup size is {} -- too small!",
                    properties.limits.max_compute_work_group_size[0]
                );
                return None;
            }

            Some(p_d)
        })
        .unwrap();

    // there's guaranteed to be a queue family with compute and transfer (and graphics)
    // even though the transfer queue family would probably be better for the asynchronous transfer I want to do
    // why overcomplicate it it'll probably be fine
    // plus i would need to like transfer the resources between queue families or something
    let qf_props = instance.get_physical_device_queue_family_properties(physical_device);
    let qf_index = qf_props
        .iter()
        .enumerate()
        .find(|(_, props)| {
            props
                .queue_flags
                .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER)
        })
        .unwrap()
        .0 as u32;
    let priorities = [1.0];
    let queue_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(qf_index)
        .queue_priorities(&priorities);

    let device_extension_names_raw = [vk::ExtShaderAtomicFloatFn::name().as_ptr()];
    let features = vk::PhysicalDeviceFeatures::builder().build();
    let mut atomic_floats = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::builder()
        .shader_shared_float32_atomics(true)
        .shader_shared_float32_atomic_add(true)
        .build();
    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_extension_names(&device_extension_names_raw)
        .enabled_features(&features)
        .push_next(&mut atomic_floats);
    let device = instance
        .create_device(physical_device, &device_create_info, None)
        .unwrap();

    let queue = device.get_device_queue(qf_index, 0);

    let info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(qf_index)
        .build();
    let command_pool = device.create_command_pool(&info, ALLOCATOR).unwrap();

    let memory_properties = instance.get_physical_device_memory_properties(physical_device);

    GPUState {
        entry,
        instance,
        debug_utils_loader,
        debug_callback,
        device,
        queue,
        qf_index,
        memory_properties,
        command_pool,
    }
}

const ALLOCATOR: Option<&'static vk::AllocationCallbacks> = None;

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if !callback_data.p_message_id_name.is_null() {
        CStr::from_ptr(callback_data.p_message_id_name)
            .to_str()
            .unwrap()
    } else {
        ""
    };
    let message = if !callback_data.p_message.is_null() {
        CStr::from_ptr(callback_data.p_message).to_str().unwrap()
    } else {
        ""
    };

    println!(
        "{:?}, {:?} [{} ({})] : {}",
        message_severity, message_type, message_id_name, message_id_number, message,
    );

    vk::FALSE
}
