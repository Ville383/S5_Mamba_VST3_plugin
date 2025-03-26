#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginAudioProcessor::AudioPluginAudioProcessor()
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
					 ),
	   parameters (*this, nullptr, "PARAMETERS", createParameterLayout())
{
	// Use MemoryInputStream to read the embedded JSON data
    juce::MemoryInputStream jsonStream (BinaryData::model_weights_json,
                                        BinaryData::model_weights_jsonSize,
                                        false);
    
    // Parse JSON
    jsonModel = nlohmann::json::parse(jsonStream.readEntireStreamAsString().toStdString());

    // set FiLM weights:
    FiLM_weights1 = jsonModel["layers"][0]["weights"][0].get<std::vector<std::vector<float>>>();
    FiLM_bias1    = jsonModel["layers"][0]["weights"][1].get<std::vector<float>>();
    FiLM_weights2 = jsonModel["layers"][1]["weights"][0].get<std::vector<std::vector<float>>>();
    FiLM_bias2    = jsonModel["layers"][1]["weights"][1].get<std::vector<float>>();
    FiLM_weights3 = jsonModel["layers"][2]["weights"][0].get<std::vector<std::vector<float>>>();
    FiLM_bias3    = jsonModel["layers"][2]["weights"][1].get<std::vector<float>>();
    
	// in_proj (1 -> d_model)
    weights0 = jsonModel["layers"][3]["weights"][0].get<std::vector<std::vector<float>>>();

    // S5_Mamba parameters:
    for (int i = 0; i < 8; ++i)
    {
        in_proj_W[i] = jsonModel["layers"][i + 4]["parameters"]["mamba"]["in_proj"]["weights"].get<std::vector<std::vector<float>>>();
        A_real[i]    = jsonModel["layers"][i + 4]["parameters"]["mamba"]["A_real"].get<std::vector<float>>();
        A_imag[i]    = jsonModel["layers"][i + 4]["parameters"]["mamba"]["A_imag"].get<std::vector<float>>();
        B_real[i]    = jsonModel["layers"][i + 4]["parameters"]["mamba"]["B_real"].get<std::vector<std::vector<float>>>();
        B_imag[i]    = jsonModel["layers"][i + 4]["parameters"]["mamba"]["B_imag"].get<std::vector<std::vector<float>>>();
        C_real[i]    = jsonModel["layers"][i + 4]["parameters"]["mamba"]["C_real"].get<std::vector<std::vector<float>>>();
        C_imag[i]    = jsonModel["layers"][i + 4]["parameters"]["mamba"]["C_imag"].get<std::vector<std::vector<float>>>();
        D[i]         = jsonModel["layers"][i + 4]["parameters"]["mamba"]["D"].get<std::vector<float>>();
        inv_dt[i]    = jsonModel["layers"][i + 4]["parameters"]["mamba"]["inv_dt"].get<std::vector<float>>();
        out_proj_W[i]= jsonModel["layers"][i + 4]["parameters"]["mamba"]["out_proj"]["weights"].get<std::vector<std::vector<float>>>();
        norm[i]      = jsonModel["layers"][i + 4]["parameters"]["norm"]["weight"].get<std::vector<float>>();
        eps[i]       = jsonModel["layers"][i + 4]["parameters"]["norm"]["eps"].get<float>();
    }
    
    // out_proj (d_model -> 1)
    weights1 = jsonModel["layers"][8 + 4]["weights"][0].get<std::vector<std::vector<float>>>();
	
	// set model weights
	filmgenerator.setWeights(FiLM_weights1, FiLM_bias1, FiLM_weights2, FiLM_bias2, FiLM_weights3, FiLM_bias3);
	
	in_proj.setWeights(weights0);
	
	for(int i = 0; i < 8; ++i)
	{
		mamba_left[i].setWeights(in_proj_W[i], out_proj_W[i], A_real[i], A_imag[i], B_real[i], B_imag[i], C_real[i], C_imag[i], D[i], inv_dt[i], norm[i], eps[i]);
		mamba_right[i].setWeights(in_proj_W[i], out_proj_W[i], A_real[i], A_imag[i], B_real[i], B_imag[i], C_real[i], C_imag[i], D[i], inv_dt[i], norm[i], eps[i]);
	}
	//std::cout << "xD" << std::endl;
	out_proj.setWeights(weights1);	
}

AudioPluginAudioProcessor::~AudioPluginAudioProcessor()
{
}

// define parameters
juce::AudioProcessorValueTreeState::ParameterLayout AudioPluginAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    
    // Create two float parameters with range -1.0f to 1.0f
    params.push_back (std::make_unique<juce::AudioParameterFloat>(
        "param1",         // Parameter ID
        "Tone",           // Parameter name (shown in the editor)
        -1.0f, 1.0f,      // Minimum and maximum values
         0.0f));          // Default value

    params.push_back (std::make_unique<juce::AudioParameterFloat>(
        "param2",         // Parameter ID
        "Gain",           // Parameter name
        -1.0f, 1.0f,      // Minimum and maximum values
         0.0f));          // Default value

    return { params.begin(), params.end() };
}



//==============================================================================
const juce::String AudioPluginAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool AudioPluginAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool AudioPluginAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool AudioPluginAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double AudioPluginAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int AudioPluginAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int AudioPluginAudioProcessor::getCurrentProgram()
{
    return 0;
}

void AudioPluginAudioProcessor::setCurrentProgram (int index)
{
    juce::ignoreUnused (index);
}

const juce::String AudioPluginAudioProcessor::getProgramName (int index)
{
    juce::ignoreUnused (index);
    return {};
}

void AudioPluginAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
    juce::ignoreUnused (index, newName);
}

//==============================================================================
void AudioPluginAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(sampleRate, samplesPerBlock);
    
    // Reset model states
    for (int i = 0; i < 8; ++i)
    {
        mamba_left[i].reset();
        mamba_right[i].reset();
    }
}


void AudioPluginAudioProcessor::releaseResources()
{
}

bool AudioPluginAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}

void AudioPluginAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused(midiMessages);
    juce::ScopedNoDenormals noDenormals;

    const int totalNumInputChannels  = getTotalNumInputChannels();
    const int totalNumOutputChannels = getTotalNumOutputChannels();
    const int numSamples = buffer.getNumSamples();

    // Clear unused channels
    for (int ch = totalNumInputChannels; ch < totalNumOutputChannels; ++ch)
        buffer.clear(ch, 0, numSamples);

    // Get parameter values and update the FiLM generator
    const float param1Val = *parameters.getRawParameterValue("param1");
    const float param2Val = *parameters.getRawParameterValue("param2");
    float c[] = { param1Val, param2Val };
	// Conidtioning calcaulted once before every processing loop
    filmgenerator.forward(c);

    // Process each channel
    for (int channel = 0; channel < totalNumInputChannels; ++channel)
    {
        float* channelData = buffer.getWritePointer(channel);
        // Process the 48 kHz signal
        // TODO: add a polyphase resampler
        if (channel == 0)
        {
            for (int n = 0; n < numSamples; ++n)
            {
                float sampleInput[] = { channelData[n] };
                in_proj.forward(sampleInput);
                mamba_left[0].forward(in_proj.outs, filmgenerator.outs);
                for (int i = 1; i < 8; ++i)
                    mamba_left[i].forward(mamba_left[i - 1].outs, filmgenerator.outs);
                channelData[n] = out_proj.forward(mamba_left[7].outs);
            }
        }
        else if (channel == 1)
        {
            for (int n = 0; n < numSamples; ++n)
            {
                float sampleInput[] = { channelData[n] };
                in_proj.forward(sampleInput);
                mamba_right[0].forward(in_proj.outs, filmgenerator.outs);
                for (int i = 1; i < 8; ++i)
                    mamba_right[i].forward(mamba_right[i - 1].outs, filmgenerator.outs);
                channelData[n] = out_proj.forward(mamba_right[7].outs);
            }
        }
    }
}


//==============================================================================
bool AudioPluginAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor()
{
    return new AudioPluginAudioProcessorEditor (*this);
}

//==============================================================================
void AudioPluginAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
    juce::ignoreUnused (destData);
}

void AudioPluginAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
    juce::ignoreUnused (data, sizeInBytes);
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AudioPluginAudioProcessor();
}
