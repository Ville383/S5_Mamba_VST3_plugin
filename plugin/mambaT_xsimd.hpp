#pragma once

#include <RTNeural/RTNeural.h>

/** Static implementation of a mamba (s5) layer. */
template <typename T, int d_modelt, int d_statet>
class MambaT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto v_d_model = RTNeural::ceil_div(d_modelt, v_size);
	static constexpr auto v_d_state = RTNeural::ceil_div(d_statet, v_size);

public:
    static constexpr auto d_model = d_modelt;
    static constexpr auto d_state = d_statet;
	static_assert(d_model % v_size == 0, "The Mamba implementation expects d_model to be a multiple of the SIMD register width.");
	static_assert(d_state % v_size == 0, "The Mamba implementation expects d_state size to be a multiple of the SIMD register width.");

    MambaT() = default;

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "mamba"; }

    constexpr bool isActivation() const noexcept { return false; }

    void reset()
	{
		for(int i = 0; i < v_d_state; ++i)
		{
			hidden_real[i] = v_type((T)0.0);
		    hidden_imag[i] = v_type((T)0.0);
		}
	}

    /** Performs forward propagation for the mamba. */
	/* input is signal from in_proj */
	/* -> in_proj -> chunk -> silu -> (ssm) -> mul -> out_proj */
    inline void forward(const v_type (&x)[v_d_model], const v_type (&c)[2][v_d_model]) noexcept
    {
		// FiLM conditioning
		for(int i = 0; i < v_d_model; ++i)
			film[i] = c[0][i] * x[i] + c[1][i];
		
		// RMS norm
		sum_RMS = v_type((T)0.0);
		for(int i = 0; i < v_d_model; ++i)
			sum_RMS += film[i] * film[i];
	
		T sum = xsimd::reduce_add(sum_RMS);
		T rms_value = std::sqrt(eps + sum / static_cast<T>(d_model));
		rms_x = v_type((T)rms_value);
		
		for(int i = 0; i < v_d_model; ++i)
			norm_x[i] = film[i] / rms_x * norm[i];
		
	    
		// in_proj linear(d_model, 2 * d_inner), modifies in_proj_outs, 4 * d_model is 2 * d_inner
		in_proj(norm_x);
		
		// chunk + silu -> use v_d_inner
        for(int i = 0; i < v_d_inner; ++i)
		{
			u[i] = in_proj_outs[i] / ((T)1.0 + xsimd::exp(-in_proj_outs[i]));
			res[i] = in_proj_outs[i + v_d_inner] / ((T)1.0 + xsimd::exp(-in_proj_outs[i + v_d_inner]));
		}

		// input: u -> h = A * h_(t-1) + B @ u
		//             y = real(C @ h) + D * u
		/*
		std::cout << "u" << std::endl;
		for(int i = 0; i < v_d_inner; ++i)
			std::cout << u[i] << std::endl;
	
		std::cout << "h" << std::endl;
		for(int i = 0; i < v_d_state; ++i)
			std::cout << hidden_real[i] << " + " << hidden_imag[i] << std::endl;
		*/
		
		ssm();
		
		// update hidden state
        for(int i = 0; i < v_d_state; ++i)
		{
		    hidden_real[i] = next_hidden_real[i];
		    hidden_imag[i] = next_hidden_imag[i];
		}

		// mul
        for(int i = 0; i < v_d_inner; ++i)
		{
			y[i] *= res[i];
		}
		
		// out_proj linear(d_inner, d_model), modifies outs
		out_proj(y);
		
		// skip connection
		for (int i = 0; i < v_d_model; ++i)
			outs[i] += x[i];
    }

    /**
     * Sets the layer weights from a given vector.
     * weights[out_size][in_size]
	 */
    void setWeights(
	const std::vector<std::vector<T>>& in_projWeights,
	const std::vector<std::vector<T>>& out_projWeights,
	const std::vector<T>& A_realWeights,
	const std::vector<T>& A_imagWeights,
	const std::vector<std::vector<T>>& B_realWeights,
	const std::vector<std::vector<T>>& B_imagWeights,
	const std::vector<std::vector<T>>& C_realWeights,
	const std::vector<std::vector<T>>& C_imagWeights,
	const std::vector<T>& D_Weights,
	const std::vector<T>& inv_dt_Weights,
	const std::vector<T>& norm_Weights,
	const float& epsilon)
    {
		// set in_proj weights
        for(int i = 0; i < d_inner_2; ++i)
        {
            for(int k = 0; k < d_model; ++k)
            {
                in_proj_w[k][i / v_size] = RTNeural::set_value(in_proj_w[k][i / v_size], i % v_size, in_projWeights[i][k]);
            }
        }
		
		// set out_proj weights
        for(int i = 0; i < d_model; ++i)
        {
            for(int k = 0; k < d_inner; ++k)
            {
                out_proj_w[k][i / v_size] = RTNeural::set_value(out_proj_w[k][i / v_size], i % v_size, out_projWeights[i][k]);
            }
        }
		
        // set A and inv_dt weights
        for(int i = 0; i < d_state; ++i)
		{
            A_real[i / v_size] = RTNeural::set_value(A_real[i / v_size], i % v_size, A_realWeights[i]);
			A_imag[i / v_size] = RTNeural::set_value(A_imag[i / v_size], i % v_size, A_imagWeights[i]);
			inv_dt[i / v_size] = RTNeural::set_value(inv_dt[i / v_size], i % v_size, inv_dt_Weights[i]);
		}
		
		// set B weights
        for(int i = 0; i < d_inner; ++i)
        {
            for(int k = 0; k < d_state; ++k)
            {
                B_real[k / v_size][i] = RTNeural::set_value(B_real[k / v_size][i], k % v_size, B_realWeights[i][k]);
				B_imag[k / v_size][i] = RTNeural::set_value(B_imag[k / v_size][i], k % v_size, B_imagWeights[i][k]);
            }
        }
		
		// set C weights
        for(int i = 0; i < d_state; ++i)
        {
            for(int k = 0; k < d_inner; ++k)
            {
                C_real[k / v_size][i] = RTNeural::set_value(C_real[k / v_size][i], k % v_size, C_realWeights[i][k]);
				C_imag[k / v_size][i] = RTNeural::set_value(C_imag[k / v_size][i], k % v_size, C_imagWeights[i][k]);
            }
        }
		
        // set D  weights
        for(int i = 0; i < d_inner; ++i)
		{
			D[i / v_size] = RTNeural::set_value(D[i / v_size], i % v_size, D_Weights[i]);
		}
		
		// set norm weights
		for(int i = 0; i < d_model; ++i)
			norm[i / v_size] = RTNeural::set_value(norm[i / v_size], i % v_size, norm_Weights[i]);	
		eps = epsilon;
		
		// softplus
		for(int i = 0; i < v_d_state; ++i)
			dt[i] = xsimd::log((T)1.0 + xsimd::exp(inv_dt[i]));
		
		// set dA and dB
		discretize_bilinear();
		
		
		/*
		std::cout << "in_proj" << std::endl;
		for(int i = 0; i < d_model; ++i)
		{
			for(int j = 0; j < v_d_inner_2; ++j)
				std::cout << in_proj_w[i][j] << std::endl;
			std::cout << std::endl;
		}
		
		std::cout << "dA" << std::endl;
		for(int i = 0; i < v_d_state; ++i)
			std::cout << dA_real[i] << " + " << dA_imag[i] << "j" << std::endl;
		
		std::cout << "dB" << std::endl;
		for(int i = 0; i < v_d_state; ++i)
		{
			for(int j = 0; j < d_inner; ++j)
				std::cout << dB_real[i][j] << " + " << dB_imag[i][j] << "j" << std::endl;	
		}
		
		std::cout << "dt" << std::endl;
		for(int i = 0; i < v_d_state; ++i)
			std::cout << dt[i] << std::endl;
		*/
    }

    v_type outs[v_d_model];


private:
	static constexpr auto d_inner = 2 * d_model;
	static constexpr auto d_inner_2 = 2 * d_inner;

	static constexpr auto v_d_model_2 = RTNeural::ceil_div(2 * d_model, v_size);
    static constexpr auto v_d_inner = RTNeural::ceil_div(d_inner, v_size);
    static constexpr auto v_d_inner_2 = RTNeural::ceil_div(d_inner_2, v_size);
	
	// model parameters
	v_type in_proj_w[d_model][v_d_inner_2];

	// s5 params
	v_type A_real[v_d_state];
	v_type A_imag[v_d_state];
	v_type B_real[v_d_state][d_inner];
	v_type B_imag[v_d_state][d_inner];
	v_type C_real[v_d_inner][d_state];
	v_type C_imag[v_d_inner][d_state];
	v_type D[v_d_inner];
	v_type inv_dt[v_d_state];

    v_type out_proj_w[d_inner][v_d_model];
	
	// RMS norm params
	v_type norm[v_d_model];
	T eps;
	v_type norm_x[v_d_model];
	v_type sum_RMS;
	v_type rms_x;
	
	// ssm hidden state params
	v_type hidden_real[v_d_state];
	v_type hidden_imag[v_d_state];
	v_type next_hidden_real[v_d_state];
	v_type next_hidden_imag[v_d_state];
	
	// FiLM paras
	v_type film[v_d_model];
	
    // forward parameters/placeholders
	v_type in_proj_outs[v_d_inner_2];
	
	v_type u[v_d_inner];
	v_type res[v_d_inner];

	v_type dt[v_d_state];
	
	v_type BL_real[v_d_state];
	v_type BL_imag[v_d_state];
	
	v_type dA_real[v_d_state];
	v_type dA_imag[v_d_state];
	v_type dB_real[v_d_state][d_inner];
	v_type dB_imag[v_d_state][d_inner];
	
	v_type Bu_elements_real[v_d_state];
	v_type Bu_elements_imag[v_d_state];
	v_type y[v_d_inner];
	
    /** Performs nn.Linear */
	// d_model -> d_inner_2
    void inline in_proj(const v_type (&ins)[v_d_model]) noexcept
    {
        //static constexpr auto v_size_inner = std::min(v_size, d_model);
		
        for(int i = 0; i < v_d_inner_2; ++i)
			in_proj_outs[i] = v_type((T)0.0);

        T scalar_in alignas(RTNEURAL_DEFAULT_ALIGNMENT)[v_size] { (T)0 };
        for(int k = 0; k < v_d_model; ++k)
        {
			ins[k].store_aligned(scalar_in);
            for(int i = 0; i < v_d_inner_2; ++i)
            {
                for(int j = 0; j < v_size; ++j)
                    in_proj_outs[i] += scalar_in[j] * in_proj_w[k * v_size + j][i];
            }
        }
    }

    /** Performs nn.Linear */
	// d_inner -> d_model
    void inline out_proj(const v_type (&ins)[v_d_inner]) noexcept
    {
        //static constexpr auto v_size_inner = std::min(v_size, d_inner);
		
        for(int i = 0; i < v_d_model; ++i)
            outs[i] = v_type((T)0.0);

        T scalar_in alignas(RTNEURAL_DEFAULT_ALIGNMENT)[v_size] { (T)0 };
        for(int k = 0; k < v_d_inner; ++k)
        {
			ins[k].store_aligned(scalar_in);
            for(int i = 0; i < v_d_model; ++i)
            {
                for(int j = 0; j < v_size; ++j)
                    outs[i] += scalar_in[j] * out_proj_w[k * v_size + j][i];
            }
        }
    }
	
    /** Performs bilinear transformation */
    void inline discretize_bilinear() noexcept
    {
		// discretize_bilinear (A)
	    for(int i = 0; i < v_d_state; ++i)
		{
            auto dt_div_2 = dt[i] / (T)2.0;
            // 1/(c + di) -> c / (c^2 + d^2) + i(-d / (c^2 + d^2))
			auto denom_c = (T)1.0 - dt_div_2 * A_real[i];
			auto denom_d = dt_div_2 * A_imag[i]; // no need for the minus sign
			auto denom = denom_c * denom_c + denom_d * denom_d;

			// Compute the complex reciprocal 1/(denom_c + i*denom_d)
			BL_real[i] = denom_c / denom;
			BL_imag[i] = denom_d / denom; // note the minus sign cancels the one in denom_d

			// Now compute 1 + dt_div_2 * A
			auto c = (T)1.0 + dt_div_2 * A_real[i];
			auto d = dt_div_2 * A_imag[i];

			// Compute dA = BL * (1 + dt_div_2 * A) using complex multiplication:
			dA_real[i] = BL_real[i] * c - BL_imag[i] * d;
			dA_imag[i] = BL_real[i] * d + BL_imag[i] * c;
		}
		
		// discretize_bilinear (B)
		for (int i = 0; i < v_d_state; ++i)
		{
			auto a = BL_real[i] * dt[i];
			auto b = BL_imag[i] * dt[i];
			// better optimization? (no need as only used in setWeights function)
			for (int j = 0; j < d_inner; ++j)
			{
				dB_real[i][j] = a * B_real[i][j] - b * B_imag[i][j];
				dB_imag[i][j] = a * B_imag[i][j] + b * B_real[i][j];
			}
		}
    }
	
	// output y[v_d_inner]
	void inline ssm() noexcept
	{	
	    // dB @ u
		for(int i = 0; i < v_d_state; ++i)
		{
            Bu_elements_real[i] = v_type((T)0.0);
		    Bu_elements_imag[i] = v_type((T)0.0);
		}
		
		//v_type u[v_d_inner];
	    //v_type u[v_d_inner];
		//v_type dB_real[v_d_state][d_inner];
		//v_type dB_imag[v_d_state][d_inner];
		T scalar_u alignas(RTNEURAL_DEFAULT_ALIGNMENT)[v_size] { (T)0 };
        for(int i = 0; i < v_d_inner; ++i)
        {
			u[i].store_aligned(scalar_u);
            for(int j = 0; j < v_d_state; ++j)
            {
				for(int k = 0; k < v_size; ++k)
				{
					Bu_elements_real[j] += dB_real[j][i * v_size + k] * scalar_u[k];
					Bu_elements_imag[j] += dB_imag[j][i * v_size + k] * scalar_u[k];
				}
            }
        }
		
		/*
		std::cout << "Bu_elements" << std::endl;
		for(int j = 0; j < v_d_state; ++j)
			std::cout << Bu_elements_real[j] << " + " << Bu_elements_imag[j] << "j" << std::endl;
		*/
		
		// h * dA + Bu_elements
        for(int i = 0; i < v_d_state; ++i)
        {
			auto a = hidden_real[i];
			auto b = hidden_imag[i];
			auto c = dA_real[i];
			auto d = dA_imag[i];
			next_hidden_real[i] = (a * c - b * d) + Bu_elements_real[i];
			next_hidden_imag[i] = (a * d + b * c) + Bu_elements_imag[i];
        }

		/*
		std::cout << "next_hidden" << std::endl;
		for(int j = 0; j < v_d_state; ++j)
			std::cout << next_hidden_real[j] << " + " << next_hidden_imag[j] << "j" << std::endl;
		*/
		
        // (C @ next_hidden).real + D * u
		for(int i = 0; i < v_d_inner; ++i)
			y[i] = D[i] * u[i];
		
		T scalar_h_real alignas(RTNEURAL_DEFAULT_ALIGNMENT)[v_size] { (T)0 };
		T scalar_h_imag alignas(RTNEURAL_DEFAULT_ALIGNMENT)[v_size] { (T)0 };
        for(int i = 0; i < v_d_state; ++i)
        {
			next_hidden_real[i].store_aligned(scalar_h_real);
			next_hidden_imag[i].store_aligned(scalar_h_imag);
            for(int j = 0; j < v_d_inner; ++j)
            {
				for(int k = 0; k < v_size; ++k)
					y[j] += scalar_h_real[k] * C_real[j][i * v_size + k] - scalar_h_imag[k] * C_imag[j][i * v_size + k];
            }
        }
		
		/*
		std::cout << "y" << std::endl;
		for(int j = 0; j < v_d_inner; ++j)
			std::cout << y[j] << std::endl;
		*/
	}
};
