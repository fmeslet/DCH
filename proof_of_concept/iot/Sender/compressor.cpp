#include "compressor.h"
#include "frequency_table.h"
#include "arithmetic_coder.h"



/*
 * CONSTRUCTOR
 */

                           
Compressor::Compressor(int alphabetS, 
                       ContextMapper* contextM,
                       int timeS,
                       int sizePacketB,
                       int sizePacketH,
                       int sizeC):
  alphabetSize(alphabetS),
  contextMapper(contextM),
  timesteps(timeS),
  sizePacketBin(sizePacketB),
  sizePacketHex(sizePacketH),
  sizeContext(sizeC)
{

  sizeDataCompressBin = 0;

  sizeDataCompressHex = sizeof(
    dataCompressHex)/sizeof(dataCompressHex[0]);
  for (int i=0; i<sizeDataCompressHex; i++)
  {
    dataCompressHex[i] = 0;
  }
  sizeDataCompressHex = 0;
};


/*
 * GETTER
 */


uint8_t* Compressor::getDataCompressBin()
{
  return dataCompressBin;
}

int Compressor::getSizeDataCompressBin()
{
  return sizeDataCompressBin;
}

uint8_t* Compressor::getDataCompressHex()
{
  return dataCompressHex;
}

int Compressor::getSizeDataCompressHex()
{
  return sizeDataCompressHex;
}


/*
 * METHODS
 */


void Compressor::setCumul(float* freqs, 
                          uint64_t* cumul)
{
  for(int i=0; i<alphabetSize+1; i++)
  {    
    if (i > 0)
    {
      cumul[i] = round(freqs[i-1]*10000000.) + 1 + cumul[i-1]; // 10000000 if 64 bits... // 10000
    } 
    else
    {
      cumul[i] = 0; 
    }
  }
  Serial.print("[DEBUG][Compressor::setCumul] cumul : ");
  Serial.print("[ "); 
  for(int i = 0; i < alphabetSize+1-1; i++) { 
    Serial.print(cumul[i]);
    Serial.print(", ");
  }
  // Print last element
  Serial.print(cumul[alphabetSize+1-1]);
  Serial.println(" ]");
}


/*
 * OTHERS
 */


void Compressor::convertBinToHex(
            uint8_t* arrayBin,
            int sizeArrayBin,
            uint8_t* arrayHex)
{
  int value = 0;
  int index = 0;

  for (int i=0; i<sizeArrayBin; i=i+8)
  {
    for (int j=0; j<8; j++)
    {
      value = value + pow(2, 7-j)*arrayBin[i+j];
    }
    arrayHex[index] = value;
    value = 0;
    index++;
  }
}

 
void Compressor::convertHexToBin(
            uint8_t* arrayHex,
            int sizeArrayHex,
            uint8_t* arrayBin)
{  
  uint8_t mask;
  uint8_t value;
  int index = 0;

  for (int i=0; i<sizeArrayHex; i++)
  {
    for (int j=0; j<8; j++)
    {
      mask = pow(2, 7-j);
      value = mask & arrayHex[i];
      
      if (value == mask)
      {
        arrayBin[(i+index)+j] = 1; 
      }
      else
      {
        arrayBin[(i+index)+j] = 0; 
      }
    }
    index = index + 7;
  }
}


void Compressor::compress(uint8_t* dataHex,
                          int sizeDataHex,
                          uint8_t* keyArray,
                          int sizeKeyArray)
{
    // Concat data et conversion en binaire
    int sizeContextBin = sizePacketBin * sizeContext;
    int contextSize = contextMapper->getContextSize(
      keyArray, sizeKeyArray);
    int sizeDataBin = sizeDataHex * 8;

    // Init array to zero
    uint8_t* dataBin = new uint8_t[sizeDataBin];
    for (int i=0; i<sizeDataBin; i++) { dataBin[i] = 0; }
    
    this->convertHexToBin(
      dataHex, sizeDataHex, dataBin);
      
    uint8_t* dataContextBin;
    dataContextBin = contextMapper->getDataContext(
      keyArray, sizeKeyArray); //dataBin, sizeDataBin);
    
    /*this->dataProcessing(contextHex,
                         dataHex,
                         sizeDataHex,
                         dataContextBin,
                         dataBin);*/

    // Create input array
    int sizeDataInput = (sizePacketBin*sizeContext)+timesteps;
    uint8_t* dataInput = new uint8_t[sizeDataInput];
    for (int i=0; i<sizeDataInput; i++) 
    {
      if (i < sizeContextBin)
      {
        dataInput[i] = dataContextBin[i];
      }
      else
      {
        dataInput[i] = 0;
      }
    }
    
    // Init context part
    FrequencyTable* frequencyTable = \
            new FrequencyTable();
    ArithmeticEncoder* arithmeticEncoder = \
            new ArithmeticEncoder(32); // 16

    // Init freqs and cumul array
    // uint8_t* cumul = new uint8_t[alphabetSize+1];
    uint64_t* cumul = new uint64_t[alphabetSize+1];
    float* freqs = new float[alphabetSize];
    for (int i=0; i<alphabetSize; i++)
    {
      freqs[i] = 1./alphabetSize;
    }
    this->setCumul(freqs, cumul);

    // Init with uniform law
    uint8_t initVal;
    for(int i=0; i<timesteps; i++)
    {
      initVal = dataBin[i];
      arithmeticEncoder->write(
        cumul, initVal, alphabetSize);
    }

    // Compress data
    // -timesteps for input block and -1 for last bit to predict
    for(int i=0; i<sizeDataBin-timesteps; i++)
    {
      // Update input part
      for (int j=0; j<timesteps; j++)
      {
        dataInput[sizePacketBin*sizeContext+j] = dataBin[i+j];
      }

      // Free memory
      delete [] freqs;

      // Extract value and get frequency
      if (i == 0)
      {
        // Extract value and get frequency
        freqs = frequencyTable->getFrequency(
          dataInput, contextSize, i);
      }
      else 
      {
        freqs = frequencyTable->nextFrequency(
          dataInput, contextSize, i);
      }
      
      this->setCumul(freqs, cumul);

      arithmeticEncoder->write(
        cumul, dataBin[(timesteps-1)+i+1], alphabetSize);
        
    }

    // Finish encoding
    arithmeticEncoder->finish();

    dataCompressBin = arithmeticEncoder->getDataCompressBin();
    sizeDataCompressBin = arithmeticEncoder->getSizeDataCompressBin();

    // Convertion to Hex
    sizeDataCompressHex = ceil(sizeDataCompressBin / 8.); // /!\ IL FAUT ARRONDIR A 8 PRES !
    this->convertBinToHex(
      dataCompressBin, sizeDataCompressBin, dataCompressHex);

    // Free memory
    //delete [] dataContextBin;
    delete [] dataBin;
    delete [] dataInput;
    delete [] cumul;
    delete [] freqs;
    delete frequencyTable;
    delete arithmeticEncoder;
}


/*void Compressor::decompress(uint8_t** contextHex,
                            int sizeContext,
                            uint8_t* dataHex,
                            int sizeArrayHex)
{

}*/
