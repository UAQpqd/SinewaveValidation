/* Methods Comparison:
 * This application will create a
 * synthetic power signal with a
 * notable sag and a flicker
 * by the end of the time window.
 * Detection accuracy is compared
 * between the proposed method and
 * the one at GA - PSO.
 */
#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <deque>
#include <map>
#include <thread>
#include <future>
#include <boost/compute.hpp>
#include <boost/chrono.hpp>
#include "synthSignal/Signal.hpp"
#include "synthSignal/SineWaveform.hpp"
#include "minusDarwin/Solver.hpp"
#include <Utility.hpp>

#include "ProposedMethodology.hpp"

const float signalTime = 8.0f;
const unsigned int sps = 8000;
const float a = 127.0f*sqrt(2);
const float omega = 2.0f*M_PI*60.0f*0.96f;
const float phi = M_PI;

std::vector<float> *createDisturbedSagSignal();

int main() {
    std::cout << "[Methods Comparison] Initializing test" << std::endl;
    // Create the sag signal
    auto signalData = createDisturbedSagSignal();
    std::vector<float> harmonicFactors = {3.0f, 5.0f, 7.0f};
    ProposedMethodology proposedMethodology(
            sps,config.epsilon,
            60.0f,
            60.0f*0.10f,
            harmonicFactors
    );
    proposedMethodology.run(signalData);
    ProposedMethodology fixedMethodology(
            sps,config.epsilon,
            60.0f,
            0.00005f,
            harmonicFactors
    );
    fixedMethodology.run(signalData);
    return 0;
}

std::vector<float> *createDisturbedSagSignal() {
    SynthSignal::Signal signalModel;
    SynthSignal::Interpolation interpolation({}, SynthSignal::InterpolationType::LINEAR);
    interpolation.addPoint(0.0f, 1.0f);
    interpolation.addPoint(signalTime, 1.0f);
    SynthSignal::Interpolation frequencyVariation({}, SynthSignal::InterpolationType::LINEAR);
    frequencyVariation.addPoint(0.0f, 1.0f);
    frequencyVariation.addPoint(signalTime, 1.0f);
    SynthSignal::Interpolation sagInterpolation({}, SynthSignal::InterpolationType::LINEAR);
    sagInterpolation.addPoint(0.0f, 0.0f);
    sagInterpolation.addPoint(signalTime-1.0f, 0.0f);
    sagInterpolation.addPoint(signalTime-0.90f, 1.0f);
    sagInterpolation.addPoint(signalTime, 1.0f);
    auto wf = new SynthSignal::SineWaveform(
            a,
            omega,
            phi,
            frequencyVariation
    );
    auto wf2 = new SynthSignal::SineWaveform(
            a*0.1f,
            2.0f*(float)M_PI*3.0f,
            phi+0.2f,
            frequencyVariation
    );
    auto harmonic3 = new SynthSignal::SineWaveform(
            a*0.1f,
            omega*3.0f,
            phi,
            frequencyVariation
    );
    auto harmonic5 = new SynthSignal::SineWaveform(
            a*0.05f,
            omega*5.0f,
            phi,
            frequencyVariation
    );
    auto harmonic7 = new SynthSignal::SineWaveform(
            a*0.025f,
            omega*7.0f,
            phi,
            frequencyVariation
    );
    auto sag = new SynthSignal::SagSwellInterruption(0.5f);
    signalModel.addEvent(wf, interpolation);
    signalModel.addEvent(wf2, interpolation);
    signalModel.addEvent(harmonic3, interpolation);
    signalModel.addEvent(harmonic5, interpolation);
    signalModel.addEvent(harmonic7, interpolation);
    signalModel.addEvent(sag, sagInterpolation);

    auto signal = signalModel.gen(signalTime, sps);
    signalModel.toCsv("rawSagSignal.csv");
    return signal;
}

