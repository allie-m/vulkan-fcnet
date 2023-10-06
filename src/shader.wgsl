// we're using storage for xbatch/ybatch cause uniform has annoying alignment requirements
@group(0) @binding(0) var<storage, read> xbatch: array<f32>; // batch_size x 28*28
@group(0) @binding(1) var<storage, read> ybatch: array<u32>; // batch_size

@group(1) @binding(0) var<storage, read_write> w: array<f32>; // 28*28 x 10
@group(1) @binding(1) var<storage, read_write> b: array<f32>; // 10

// fixed-radix number
// from testing i think i've found a sweet spot for radix values
// so interpret these integers as (sign bit)(6 bits).(25 bits)
@group(2) @binding(0) var<storage, read_write> scores: array<atomic<u32>>; // batch_size x 10

// @group(3) @binding(0) var<storage, write> loss: array<f32>;

// @group(4) @binding(0) var<storage, write> gradients: array<f32>; // batch_size x 10

// var<push_constant> current_iteration: u32;

// xbatch @ w + b -- final shape is batch_size x 10
// e^x for each element
// divide each row by its sum
// this shader is to be invoked with workgroups (batch_size, 10, 28*28)
@compute
@workgroup_size(1, 1, 1)
fn calculate_scores(@builtin(global_invocation_id) gid: vec3<u32>) {
    let val: f32 = xbatch[gid.x * 28u * 28u + gid.z] * w[gid.z * 10u + gid.y] + b[gid.y];
    if (val == 0.0) {
        return;
    }
    // converting val into a fixed radix number
    let bits: u32 = bitcast<u32>(val);
    let unshifted_e: i32 = i32(((bits >> 23u) & 0xffu));
    var mantissa: u32;
    let sign = bits >> 31u;
    if (unshifted_e == 0) {
        mantissa = (bits & 0x7fffffu) << 1u;
    } else {
        mantissa = (bits & 0x7fffffu) | 0x800000u;
    }
    let exponent = unshifted_e - 127 + 23;

    var radix_num: u32;
    if (-exponent > 25) {
        radix_num = mantissa >> u32(-25 - exponent);
    } else {
        radix_num = mantissa << u32(25 + exponent);
    }
    radix_num |= (sign << 31u);

    atomicAdd(&scores[gid.x * 10u + gid.y], radix_num);
}

// TODO
// @compute
// @workgroup_size(32, 1, 1)
// fn calculate_gradients(@builtin(global_invocation_id) gid: vec3<u32>) {
//     //
// }

// @compute
// @workgroup_size(28, 1, 1)
// fn backpropagate(@builtin(global_invocation_id) gid: vec3<u32>) {
//     //
// }
