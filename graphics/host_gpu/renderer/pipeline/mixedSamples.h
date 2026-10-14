#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MIXEDSAMPLES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MIXEDSAMPLES_H_

// ---------------------------------------------------------------------------
// Mixed color/depth sample counts (PS5: DB at N_d samples, CB at N_c samples).
//
// The PS5 GPU (RDNA2) drives its depth block and its color block at independent
// sample rates: a render target can be 1x while the depth buffer bound to the
// same pass is 2x/4x. Vulkan core cannot express that (one
// VkPipelineMultisampleStateCreateInfo::rasterizationSamples for the whole
// pass), which is why pipelineCache.cpp used to abort with
//   "mixed color/depth sample counts are unsupported: 1 and 2".
//
// This header holds the *pure* part of the fix: sample-count planning and the
// coverage/resolve arithmetic. It deliberately includes nothing but <array> and
// <cstdint> so that it can be unit-tested outside of the emulator (see
// tests/MixedSamplesTests.cpp).
// ---------------------------------------------------------------------------

#include <array>
#include <cstdint>

namespace Libs::Graphics {
namespace MixedSamples {

// Must stay in sync with RENDER_COLOR_ATTACHMENTS_MAX (renderTarget.h).
static constexpr uint32_t kColorSlots = 8u;
static constexpr uint32_t kMaxSamples = 8u;

// Which vendor extension exposes mixed attachment samples on this host.
enum class Vendor : uint8_t {
	kNone = 0, // no host support: the pass has to be degraded
	kAmd  = 1, // VK_AMD_mixed_attachment_samples
	kNv   = 2, // VK_NV_framebuffer_mixed_samples (+ VK_NV_coverage_reduction_mode)
};

// What the renderer finally does with the pass.
enum class Strategy : uint8_t {
	kNative    = 0, // every attachment keeps its own sample count (exact PS5 behaviour)
	kDropDepth = 1, // no host support: rasterise at the color rate, drop the depth attachment
	kDropColor = 2, // no host support: rasterise at the highest rate, drop the mismatching targets
};

struct AttachmentSamples {
	uint32_t slot    = 0; // color target slot (0..kColorSlots-1)
	uint32_t samples = 0; // 1, 2, 4 or 8
};

struct Plan {
	uint32_t                            rasterization = 1; // pipeline rasterizationSamples == max(N_c, N_d)
	std::array<uint32_t, kColorSlots>   color {};          // per slot, 0 == slot unused
	uint32_t                            depth = 0;         // 0 == no depth attachment
	bool                                mixed = false;     // any attachment differs from rasterization
	Strategy                            strategy = Strategy::kNative;

	[[nodiscard]] constexpr bool HasColor() const noexcept {
		for (const auto samples: color) {
			if (samples != 0) return true;
		}
		return false;
	}
	[[nodiscard]] constexpr bool HasDepth() const noexcept { return depth != 0; }
	// Sample count a slot declared as unused must advertise: the pipeline needs a
	// valid VkSampleCountFlagBits for every attachment index < colorAttachmentCount.
	[[nodiscard]] constexpr uint32_t SlotSamples(uint32_t slot) const noexcept {
		return color[slot] != 0 ? color[slot] : rasterization;
	}
};

constexpr uint32_t SampleCountLog2(uint32_t samples) noexcept {
	// 1 -> 0, 2 -> 1, 4 -> 2, 8 -> 3; anything else is not a Vulkan sample count.
	return samples == 1u ? 0u : samples == 2u ? 1u : samples == 4u ? 2u : samples == 8u ? 3u : 0xffu;
}

constexpr uint32_t SampleCountFromLog2(uint32_t log2) noexcept { return 1u << log2; }

constexpr bool IsValidSampleCount(uint32_t samples) noexcept {
	return SampleCountLog2(samples) != 0xffu;
}

// Build the plan from the guest attachment descriptions.
constexpr Plan BuildPlan(const AttachmentSamples* colors, uint32_t color_count,
                         uint32_t depth_samples) noexcept {
	Plan plan {};
	uint32_t rasterization = depth_samples;
	for (uint32_t i = 0; i < color_count; i++) {
		if (colors[i].slot < kColorSlots) {
			plan.color[colors[i].slot] = colors[i].samples;
		}
		if (colors[i].samples > rasterization) {
			rasterization = colors[i].samples;
		}
	}
	plan.depth         = depth_samples;
	plan.rasterization = rasterization != 0 ? rasterization : 1u;
	for (uint32_t slot = 0; slot < kColorSlots; slot++) {
		if (plan.color[slot] != 0 && plan.color[slot] != plan.rasterization) {
			plan.mixed = true;
		}
	}
	if (plan.depth != 0 && plan.depth != plan.rasterization) {
		plan.mixed = true;
	}
	return plan;
}

// Can the host express this plan natively?
//
// Constraints that matter in practice:
//  * the depth sample count must be a non-zero multiple of every color sample count
//    (VK_NV_framebuffer_mixed_samples: "the number of samples in the depth/stencil
//     attachment is a multiple of the number of samples in the color attachments");
//  * every color attachment must share one sample count (NVIDIA hardware cannot mix
//    color attachments; AMD is unspecified, so we stay portable and require it too);
//  * the rasterization sample count is the depth one as soon as a depth is bound.
constexpr bool IsSupported(Vendor vendor, const Plan& plan) noexcept {
	if (!plan.mixed) {
		return true;
	}
	if (vendor == Vendor::kNone || !IsValidSampleCount(plan.rasterization)) {
		return false;
	}
	uint32_t color_samples = 0;
	for (const auto samples: plan.color) {
		if (samples == 0) continue;
		if (!IsValidSampleCount(samples)) return false;
		if (color_samples == 0) {
			color_samples = samples;
		} else if (color_samples != samples) {
			return false; // color attachments disagree with each other
		}
	}
	if (plan.depth != 0) {
		if (!IsValidSampleCount(plan.depth) || plan.depth != plan.rasterization) return false;
		if (color_samples != 0 && (plan.depth < color_samples || plan.depth % color_samples != 0)) {
			return false;
		}
	} else if (plan.rasterization == 0) {
		return false;
	}
	return true;
}

// No host support: keep the pass alive instead of aborting the session.
//
//  * the color attachments agree with each other  -> keep them (the image stays
//    visible), rasterise at the color rate and drop the depth attachment;
//  * otherwise                                    -> rasterise at the highest rate
//    and drop every attachment that does not match it.
constexpr Plan Degrade(const Plan& plan) noexcept {
	Plan result = plan;
	result.mixed = false;
	if (!plan.mixed) {
		result.strategy = Strategy::kNative;
		return result;
	}
	uint32_t color_samples = 0;
	bool     color_uniform = true;
	for (const auto samples: plan.color) {
		if (samples == 0) continue;
		if (color_samples == 0) {
			color_samples = samples;
		} else if (color_samples != samples) {
			color_uniform = false;
		}
	}
	if (color_uniform && color_samples != 0 && plan.depth != 0) {
		result.strategy      = Strategy::kDropDepth;
		result.rasterization = color_samples;
		result.depth         = 0;
		return result;
	}
	result.strategy      = Strategy::kDropColor;
	result.rasterization = plan.rasterization;
	for (uint32_t slot = 0; slot < kColorSlots; slot++) {
		if (result.color[slot] != 0 && result.color[slot] != result.rasterization) {
			result.color[slot] = 0;
		}
	}
	if (result.depth != 0 && result.depth != result.rasterization) {
		result.depth = 0;
	}
	return result;
}

// ---------------------------------------------------------------------------
// Coverage arithmetic.
//
// Rasterisation and the depth/stencil tests produce one coverage bit per
// *raster* sample: cov in [0, 2^N_r). The color attachment only stores N_c bits,
// so the coverage has to be reduced. Two reductions exist:
//
//   truncate (AMD / PS5, VK_AMD_mixed_attachment_samples):
//       C_color[i] = C_raster[i]                      for i < N_c
//       -> the extra depth samples are thrown away, the mapping is the identity.
//          This is what the PS5 hardware does, and what we ask NVIDIA for with
//          VK_COVERAGE_REDUCTION_MODE_TRUNCATE_NV (its default, MERGE, is an OR
//          over the group and would thicken every silhouette).
//
//   merge (NVIDIA default, VK_COVERAGE_REDUCTION_MODE_MERGE_NV):
//       C_color[i] = OR_{j in group(i)} C_raster[j]   with k = N_r / N_c
//       -> the association is implementation defined; the canonical grouping is
//          group(i) = { i*k .. i*k + k-1 }.
//
// The software fallback (render at N_r samples into a scratch color buffer and
// resolve) reproduces the merge case with a box filter, i.e. every raster sample
// of a group contributes 1/k of the color:
//
//       resolve(pixel) = sum_i  ( popcount(cov & group(i)) / k ) * C_color[i]
//       with  sum_i popcount(cov & group(i)) / k  ==  popcount(cov) / k
//       =>    resolve = (1 / popcount(cov)) * sum_{j in cov} C_raster[j]
//       i.e. the usual MSAA average restricted to the covered samples.
// ---------------------------------------------------------------------------

// Mask of the raster samples associated with one color sample.
constexpr uint32_t CoverageGroupMask(uint32_t color_sample, uint32_t raster,
                                     uint32_t color) noexcept {
	if (color == 0 || raster < color || (raster % color) != 0) {
		return 0;
	}
	const uint32_t group = raster / color;
	const uint32_t first = color_sample * group;
	return ((1u << group) - 1u) << first;
}

constexpr uint32_t ReduceCoverageTruncate(uint32_t coverage, uint32_t raster,
                                          uint32_t color) noexcept {
	(void)raster;
	return coverage & ((1u << color) - 1u);
}

constexpr uint32_t ReduceCoverageMerge(uint32_t coverage, uint32_t raster,
                                       uint32_t color) noexcept {
	if (color == 0 || raster < color) {
		return 0;
	}
	uint32_t reduced = 0;
	for (uint32_t i = 0; i < color; i++) {
		if ((coverage & CoverageGroupMask(i, raster, color)) != 0) {
			reduced |= 1u << i;
		}
	}
	return reduced;
}

// Fraction of the group covered by `coverage`: the weight a resolve/coverage
// modulation gives to color sample `color_sample`.
constexpr float ResolveWeight(uint32_t coverage, uint32_t raster, uint32_t color,
                              uint32_t color_sample) noexcept {
	const uint32_t group = (raster / color);
	const uint32_t bits  = coverage & CoverageGroupMask(color_sample, raster, color);
	uint32_t       count = 0;
	for (uint32_t j = 0; j < raster; j++) {
		if ((bits & (1u << j)) != 0) count++;
	}
	return static_cast<float>(count) / static_cast<float>(group != 0 ? group : 1u);
}

// ---------------------------------------------------------------------------
// Compact key payload: 2 bits per color slot (log2 of the sample count) plus the
// depth and rasterization counts. Fits in 20 bits, so the pipeline key and its
// hash (XXH3 over PipelineStaticParameters) stay as cheap as before.
// ---------------------------------------------------------------------------

constexpr uint32_t PackColorBits(const Plan& plan) noexcept {
	uint32_t bits = 0;
	for (uint32_t slot = 0; slot < kColorSlots; slot++) {
		const uint32_t value = plan.color[slot] == 0 ? 0u : SampleCountLog2(plan.color[slot]) + 1u;
		bits |= (value & 0xfu) << (slot * 4u);
	}
	return bits;
}

constexpr uint32_t UnpackSlotSamples(uint32_t bits, uint32_t slot) noexcept {
	const uint32_t value = (bits >> (slot * 4u)) & 0xfu;
	return value == 0 ? 0u : SampleCountFromLog2(value - 1u);
}

} // namespace MixedSamples
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MIXEDSAMPLES_H_
