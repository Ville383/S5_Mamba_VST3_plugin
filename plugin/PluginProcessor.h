#pragma once

//#include <JuceHeader.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <RTNeural/RTNeural.h>
#include "BinaryData.h"

#if RTNEURAL_USE_XSIMD
#include "mambaT_xsimd.hpp"
#include "filmgeneratorT_xsimd.hpp"
#include "custom_denseT_xsimd.hpp"
#endif

//==============================================================================
class AudioPluginAudioProcessor final : public juce::AudioProcessor
{
public:
    //==============================================================================
    AudioPluginAudioProcessor();
    ~AudioPluginAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;
	
	//get parameters
	juce::AudioProcessorValueTreeState& getParameters() { return parameters; }

private:
    //==============================================================================
    // Declare JSON model and weight variables as members:
    nlohmann::json jsonModel;
    
    std::vector<std::vector<float>> FiLM_weights1;
    std::vector<float> FiLM_bias1;
    std::vector<std::vector<float>> FiLM_weights2;
    std::vector<float> FiLM_bias2;
    std::vector<std::vector<float>> FiLM_weights3;
    std::vector<float> FiLM_bias3;
	
    std::vector<std::vector<float>> weights0;
    
    std::array<std::vector<std::vector<float>>, 8> in_proj_W;
    std::array<std::vector<float>, 8> A_real;
    std::array<std::vector<float>, 8> A_imag;
    std::array<std::vector<std::vector<float>>, 8> B_real;
    std::array<std::vector<std::vector<float>>, 8> B_imag;
    std::array<std::vector<std::vector<float>>, 8> C_real;
    std::array<std::vector<std::vector<float>>, 8> C_imag;
    std::array<std::vector<float>, 8> D;
    std::array<std::vector<float>, 8> inv_dt;
    std::array<std::vector<std::vector<float>>, 8> out_proj_W;
    std::array<std::vector<float>, 8> norm;
    std::array<float, 8> eps;
    
    std::vector<std::vector<float>> weights1; // out_proj
	
    // define static model (same model applied to both audio channels)
	// the input is manually converted to XSIMD suitable in the in_size_1_DenseT
	// the output is manually converted to T (float) in the out_size_1_DenseT and returned from the forward function
	// other models (mamba_left, mamba_right) expect the input to be XSIMD type
    in_size_1_DenseT<float, 16> in_proj; // nn.Linear(1, 16)
	std::array<MambaT<float, 16, 32>, 8> mamba_left;
	std::array<MambaT<float, 16, 32>, 8> mamba_right;
	out_size_1_DenseT<float, 16> out_proj; // nn.Linear(16, 1)
	
	// FiLM generator
	// the input is manually converted to XSIMD suitable in FiLMGeneratorT
    FiLMGeneratorT<float, 2, 16> filmgenerator; // outs -> v_type outs[2][v_d_model]
	
    // Create parameter layout (knobs with range -1.0f to 1.0f)
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    juce::AudioProcessorValueTreeState parameters;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessor)
};
