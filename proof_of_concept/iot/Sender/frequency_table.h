#ifndef FREQ_TABLE_H
#define FREQ_TABLE_H
#include <Arduino.h>

#define ALPHABET_SIZE     0x02
#define KEY_SIZE          0x06

class FrequencyTable
{
  private:
    uint16_t positionArray[3][448][8];

    uint8_t freqsArray[30400];
    uint8_t freqsCountArray[3][448];
    uint32_t freqsCountContextArray[3];

    int indexFreqsArray;
    int sumCountFreqsArray;

    // Methods 
    int convertBinToInt(uint8_t* dataBin, 
                            int sizeArrayBin);
    int computeKey(uint8_t* dataBin,
                            int context,
                            int position);
    int getIndexFreqsArray(int contextSize,
                            int position,
                            int key);
    int getIndexFreqsArray(int countFreqsArray,
                            int key);
    float* computeFrequency(int nextPosition);

  public:
    // Constructor
    FrequencyTable();

    // Others
    float* nextFrequency(uint8_t* dataBin, 
                           int contextSize,
                           int position);
    float* getFrequency(uint8_t* dataBin, 
                           int contextSize,
                           int position);
    // Others
    // 
};

#endif
