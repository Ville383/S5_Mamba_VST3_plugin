#pragma once

#include "PluginProcessor.h"

//==============================================================================
class AudioPluginAudioProcessorEditor final : public juce::AudioProcessorEditor
{
public:
    explicit AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor&);
    ~AudioPluginAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    AudioPluginAudioProcessor& processorRef;
	
    // Sliders for our two parameters
    juce::Slider param1Slider;
    juce::Slider param2Slider;

    // Attachments for binding the sliders to the parameters
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> param1Attachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> param2Attachment;
	
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};
