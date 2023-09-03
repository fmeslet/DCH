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

  //printf("[DEBUG][Compressor::convertHexToBin] arrayHex[0] : %d \n", arrayHex[0]);

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


void Compressor::compress(uint8_t* dataBin,
                          int sizeDataBin,
                          uint8_t* keyArray,
                          int sizeKeyArray)
{

    int sizeContextBin = sizePacketBin * sizeContext;
    int contextSize = contextMapper->getContextSize(
      keyArray, sizeKeyArray);

    /*printf("[DEBUG][Compressor::compress] sizeDataBin : %d \n", sizeDataBin);
    printf("[DEBUG][Compressor::compress] sizeContextBin : %d \n", sizeContextBin);
    printf("[DEBUG][Compressor::compress] sizeDataHex : %d \n", sizeDataHex);

    Serial.print("[DEBUG][Compressor::compress] dataHex : ");
    Serial.print("[ "); 
    for(int i = 0; i < sizeDataHex-1; i++) { 
      Serial.print(dataHex[i]);
      Serial.print(", ");
    }
  
    // Print last element
    Serial.print(dataHex[sizeDataHex-1]);
    Serial.println(" ]");


    Serial.print("[DEBUG][Compressor::compress] dataBin : ");
    Serial.print("[ "); 
    for(int i = 0; i < sizeDataBin-1; i++) { 
      Serial.print(dataBin[i]);
      Serial.print(", ");
    }
  
    // Print last element
    Serial.print(dataBin[sizeDataBin-1]);
    Serial.println(" ]");
    
    //printf("[DEBUG][Compressor::compress] sizeContextBin : %d \n", sizeContextBin);*/
    /*printf("[DEBUG][Compressor::compress] dataBin AFTER CONVERSION \n");

    Serial.print("[DEBUG][Compressor::compress] dataBin : ");
    Serial.print("[ "); 
    for(int i = 0; i < sizeDataBin-1; i++) { 
      Serial.print(dataBin[i]);
      Serial.print(", ");
    }
  
    // Print last element
    Serial.print(dataBin[sizeDataBin-1]);
    Serial.println(" ]");*/
      
    uint8_t* dataContextBin;
    //Serial.print("[DEBUG][Compressor::compress] CREATE dataContextBin\n");
    dataContextBin = contextMapper->getDataContext(
      keyArray, sizeKeyArray);

    /*printf("[DEBUG][Compressor::compress] sizeContextBin : %d\n", sizeContextBin);

    Serial.print("[DEBUG][Compressor::compress] dataContextBin : ");
    Serial.print("[ "); 
    for(int i = 0; i < sizeContextBin-1; i++) { 
      Serial.print(dataContextBin[i]);
      Serial.print(", ");
    }
  
    // Print last element
    Serial.print(dataContextBin[sizeContextBin-1]);
    Serial.println(" ]");*/

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

    //printf("[DEBUG][Compressor::compress] dataInput init done ! \n");
    
    // Init context part
    FrequencyTable* frequencyTable = \
            new FrequencyTable();
    ArithmeticEncoder* arithmeticEncoder = \
            new ArithmeticEncoder(32); // 16

    ///printf("[DEBUG][Compressor::compress] Init context part done ! \n");

    // Init freqs and cumul array
    uint64_t* cumul = new uint64_t[alphabetSize+1];
    float* freqs = new float[alphabetSize];
    
    //printf("[DEBUG][Compressor::compress] Init freqs and cumul array done ! \n");
    
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
      //printf("[DEBUG][Compressor::compress] i : %d // initVal : %d \n", i, initVal);
      arithmeticEncoder->write(
        cumul, initVal, alphabetSize);
    }

    //printf("[DEBUG][Compressor::compress] Set uniform law done ! \n");
    //printf("[DEBUG][Compressor::compress] sizeDataBin: %d \n", sizeDataBin);

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
        //printf("[DEBUG][Compressor::compress] Je passe dans le IF \n");
      }
      else 
      {
        freqs = frequencyTable->nextFrequency(
          dataInput, contextSize, i);
        //printf("[DEBUG][Compressor::compress] Je passe dans le ELSE \n");
      }
      
      this->setCumul(freqs, cumul);

      //printf("[DEBUG][Compressor::compress] dataBin[(timesteps-1)+i+1] : %d \n", dataBin[(timesteps-1)+i+1]);

      arithmeticEncoder->write(
        cumul, dataBin[(timesteps-1)+i+1], alphabetSize);

      //dataCompressBin = arithmeticEncoder->getDataCompressBin();
      //sizeDataCompressBin = arithmeticEncoder->getSizeDataCompressBin();
        
    }

    //printf("[DEBUG][Compressor::compress] Compress data ! \n");

    // Finish encoding
    arithmeticEncoder->finish();

    dataCompressBin = arithmeticEncoder->getDataCompressBin();
    sizeDataCompressBin = arithmeticEncoder->getSizeDataCompressBin();

    /*Serial.print("[DEBUG][Compressor::compress] dataCompressBin AFTER COMPRESS : ");
    Serial.print("[ "); 
    for(int i = 0; i < sizeDataCompressBin-1; i++) { 
      Serial.print(dataCompressBin[i]);
      Serial.print(", ");
    }
  
    // Print last element
    Serial.print(dataCompressBin[sizeDataCompressBin-1]);
    Serial.println(" ]");*/

    // Convertion to Hex
    sizeDataCompressHex = ceil(sizeDataCompressBin / 8.); // /!\ IL FAUT ARRONDIR A 8 PRES !
    this->convertBinToHex(
      dataCompressBin, sizeDataCompressBin, dataCompressHex);
    /*printf("[DEBUG][FrequencyTable::computeKey] sizeDataCompress BEFORE : %d \n", sizeDataCompressBin);
    printf("[DEBUG][FrequencyTable::computeKey] sizeDataCompress AFTER : %d \n", sizeDataCompressHex);
    printf("[DEBUG][Compressor::compress] dataCompress[10] : %d \n", dataCompress[10]);*/

    //printf("[DEBUG][Compressor::compress] Free memory : %d \n", sizeDataCompressHex);


    // Free memory
    //delete [] dataContextBin;
    //delete [] dataBin;
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
