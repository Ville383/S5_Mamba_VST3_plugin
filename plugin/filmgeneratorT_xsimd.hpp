#pragma once

#include <RTNeural/RTNeural.h>

/** Static implementation of a FiLMGenerator layer. */
template <typename T, int in_sizet, int out_sizet>
class FiLMGeneratorT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto v_in_size = RTNeural::ceil_div(in_sizet, v_size);
	static constexpr auto v_out_size = RTNeural::ceil_div(out_sizet, v_size);

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    FiLMGeneratorT()
	{
		for(int i = 0; i < v_in_size; ++i)
			v_ins[i] = v_type((T)0);
	}

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "FiLM Generator"; }

    constexpr bool isActivation() const noexcept { return false; }

    void reset() { }

    inline void forward(const T* input) noexcept
    {
		for(int i = 0; i < v_in_size; ++i)
			v_ins[i] = xsimd::load_aligned(input + i * v_size);
		
		FiLMGenerator(v_ins);
		
		/*
		std::cout << "gamma" << std::endl;
		for(int i = 0; i < v_out_size; ++i)
			std::cout << outs[0][i] << std::endl;
	
		std::cout << "beta" << std::endl;
		for(int i = 0; i < v_out_size; ++i)
			std::cout << outs[1][i] << std::endl;
		*/
    }

    /**
     * Sets the layer weights from a given vector.
     * weights[out_size][in_size]
	 */
    void setWeights(
	const std::vector<std::vector<T>>& FiLMWeights_1,
	const std::vector<T>& FiLMBias_1,
	const std::vector<std::vector<T>>& FiLMWeights_2,
	const std::vector<T>& FiLMBias_2,
	const std::vector<std::vector<T>>& FiLMWeights_3,
	const std::vector<T>& FiLMBias_3)
    {
		for(int i = 0; i < F_in_1; ++i)
        {
            for(int k = 0; k < F_c; ++k)
            {
                FiLM_w_1[k][i / v_size] = RTNeural::set_value(FiLM_w_1[k][i / v_size], i % v_size, FiLMWeights_1[i][k]);
            }
			FiLM_b_1[i / v_size] = RTNeural::set_value(FiLM_b_1[i / v_size], i % v_size, FiLMBias_1[i]);
        }
		
		for(int i = 0; i < F_in_2; ++i)
        {
            for(int k = 0; k < F_in_1; ++k)
            {
                FiLM_w_2[k][i / v_size] = RTNeural::set_value(FiLM_w_2[k][i / v_size], i % v_size, FiLMWeights_2[i][k]);
            }
			FiLM_b_2[i / v_size] = RTNeural::set_value(FiLM_b_2[i / v_size], i % v_size, FiLMBias_2[i]);
        }

		for(int i = 0; i < d_model_2; ++i)
        {
            for(int k = 0; k < F_in_2; ++k)
            {
                FiLM_w_3[k][i / v_size] = RTNeural::set_value(FiLM_w_3[k][i / v_size], i % v_size, FiLMWeights_3[i][k]);
            }
			FiLM_b_3[i / v_size] = RTNeural::set_value(FiLM_b_3[i / v_size], i % v_size, FiLMBias_3[i]);
        }
    }

	v_type outs[2][v_out_size];

private:
	// FiLM generator
	static constexpr auto F_c = in_size;
	static constexpr auto F_in_1 = (int)16;
	static constexpr auto F_in_2 = (int)32;
	static constexpr auto d_model = out_size;
	static constexpr auto d_model_2 = 2 * d_model;

	static constexpr auto v_F_c = RTNeural::ceil_div(F_c, v_size);
	static constexpr auto v_F_in_1 = RTNeural::ceil_div(F_in_1, v_size);
	static constexpr auto v_F_in_2 = RTNeural::ceil_div(F_in_2, v_size);
	static constexpr auto v_d_model = RTNeural::ceil_div(d_model, v_size);
	static constexpr auto v_d_model_2 = RTNeural::ceil_div(d_model_2, v_size);
	
	v_type v_ins[v_in_size];
	
	// FiLM generator params
	v_type FiLM_w_1[F_c][v_F_in_1];
	v_type FiLM_b_1[v_F_in_1];
	v_type FiLM_w_2[F_in_1][v_F_in_2];
	v_type FiLM_b_2[v_F_in_2];
	v_type FiLM_w_3[F_in_2][v_d_model_2];
	v_type FiLM_b_3[v_d_model_2];
	
	v_type FiLM_1_out[v_F_in_1];
	v_type FiLM_2_out[v_F_in_2];
	v_type gamma_beta[v_d_model_2];
	
	v_type gamma[v_d_model];
	v_type beta[v_d_model];
	
	void inline FiLMGenerator(const v_type (&ins)[v_F_c]) noexcept
    {
		// 1st layer
        static constexpr auto v_size_inner_1 = std::min(v_size, F_c);
		
        for(int i = 0; i < v_F_in_1; ++i)
			FiLM_1_out[i] = FiLM_b_1[i];

        T scalar_in_1 alignas(RTNEURAL_DEFAULT_ALIGNMENT)[v_size] { (T)0 };
        for(int k = 0; k < v_F_c; ++k)
        {
			ins[k].store_aligned(scalar_in_1);
            for(int i = 0; i < v_F_in_1; ++i)
            {
                for(int j = 0; j < v_size_inner_1; ++j)
                    FiLM_1_out[i] += scalar_in_1[j] * FiLM_w_1[k * v_size + j][i];
            }
        }	
		
		//ReLU
        for(int i = 0; i < v_F_in_1; ++i)
			FiLM_1_out[i] = xsimd::max(FiLM_1_out[i], v_type((T)0));
		
		// 2nd layer
        static constexpr auto v_size_inner_2 = std::min(v_size, F_in_1);
		
        for(int i = 0; i < v_F_in_2; ++i)
			FiLM_2_out[i] = FiLM_b_2[i];

        T scalar_in_2 alignas(RTNEURAL_DEFAULT_ALIGNMENT)[v_size] { (T)0 };
        for(int k = 0; k < v_F_in_1; ++k)
        {
			FiLM_1_out[k].store_aligned(scalar_in_2);
            for(int i = 0; i < v_F_in_2; ++i)
            {
                for(int j = 0; j < v_size_inner_2; ++j)
                    FiLM_2_out[i] += scalar_in_2[j] * FiLM_w_2[k * v_size + j][i];
            }
        }

		//ReLU
        for(int i = 0; i < v_F_in_2; ++i)
			FiLM_2_out[i] = xsimd::max(FiLM_2_out[i], v_type((T)0));

		// 3rd layer
        static constexpr auto v_size_inner_3 = std::min(v_size, F_in_2);
		
        for(int i = 0; i < v_d_model_2; ++i)
			gamma_beta[i] = FiLM_b_3[i];

        T scalar_in_3 alignas(RTNEURAL_DEFAULT_ALIGNMENT)[v_size] { (T)0 };
        for(int k = 0; k < v_F_in_2; ++k)
        {
			FiLM_2_out[k].store_aligned(scalar_in_3);
            for(int i = 0; i < v_d_model_2; ++i)
            {
                for(int j = 0; j < v_size_inner_3; ++j)
                    gamma_beta[i] += scalar_in_3[j] * FiLM_w_3[k * v_size + j][i];
            }
        }
		
		//chunk
        for(int i = 0; i < v_d_model; ++i)
		{
			outs[0][i] = gamma_beta[i]; // gamma
			outs[1][i] = gamma_beta[i + v_d_model]; // beta
		}
    }
};
