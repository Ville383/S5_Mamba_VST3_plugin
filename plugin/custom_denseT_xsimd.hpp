#pragma once

#include <RTNeural/RTNeural.h>

/**
 * Static implementation of a fully-connected (dense) layer,
 * optimized for in_size=1.
 */
template <typename T, int out_sizet>
class in_size_1_DenseT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto v_out_size = RTNeural::ceil_div(out_sizet, v_size);

public:
    static constexpr auto in_size = 1;
    static constexpr auto out_size = out_sizet;

    in_size_1_DenseT()
    {
        for(int i = 0; i < v_out_size; ++i)
            weights[i] = v_type((T)0.0);

        for(int i = 0; i < v_out_size; ++i)
            outs[i] = v_type((T)0.0);
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "in_size = 1 dense"; }

    /** Returns false since dense is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Reset is a no-op, since Dense does not have state. */
    RTNEURAL_REALTIME void reset() { }

    /** Performs forward propagation for this layer (no bias). */
	// excepts that the input is of length one!!!
    RTNEURAL_REALTIME inline void forward(const T* input) noexcept
    {
        const v_type in = ((v_type)input[0]).get(0);
        for(int i = 0; i < v_out_size; ++i)
            outs[i] = in * weights[i];
    }

    /**
     * Sets the layer weights from a given vector.
     *
     * The dimension of the weights vector must be
     * weights[out_size][in_size]
     */
    RTNEURAL_REALTIME void setWeights(const std::vector<std::vector<T>>& newWeights)
    {
        for(int i = 0; i < out_size; ++i)
            weights[i / v_size] = RTNeural::set_value(weights[i / v_size], i % v_size, newWeights[i][0]);
    }

    v_type outs[v_out_size];

private:
    v_type weights[v_out_size];
};

/**
 * Static implementation of a fully-connected (dense) layer,
 * optimized for out_size=1.
 */
template <typename T, int in_sizet>
class out_size_1_DenseT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto v_in_size = RTNeural::ceil_div(in_sizet, v_size);
	static constexpr auto v_out_size = RTNeural::ceil_div(1, v_size);

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = 1;

    out_size_1_DenseT()
    {
        for(int i = 0; i < v_in_size; ++i)
            weights[i] = v_type((T)0.0);

        outs[0] = v_type((T)0.0);
    }

    std::string getName() const noexcept { return "dense"; }
    constexpr bool isActivation() const noexcept { return false; }

    RTNEURAL_REALTIME void reset() { }

    RTNEURAL_REALTIME inline T forward(const v_type (&ins)[v_in_size]) noexcept
    {
        v_type y {};
        for(int k = 0; k < v_in_size; ++k)
            y += ins[k] * weights[k];

        outs[0] = v_type(xsimd::reduce_add(y));

		return (T)outs[0].get(0);
    }

    RTNEURAL_REALTIME void setWeights(const std::vector<std::vector<T>>& newWeights)
    {
        for(int i = 0; i < out_size; ++i)
        {
            for(int k = 0; k < in_size; ++k)
            {
                auto idx = k / v_size;
                weights[idx] = RTNeural::set_value(weights[idx], k % v_size, newWeights[i][k]);
            }
        }
    }

    v_type outs[1];

private:
    v_type weights[v_in_size];
};