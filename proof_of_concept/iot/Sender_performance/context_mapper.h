#ifndef CONTEXT_MAPPER_H
#define CONTEXT_MAPPER_H

#include <Arduino.h>
#include <map>


class ContextMapper
{
  private:
    std::map<int, uint8_t*> dataContext;
    std::map<int, int> contextSize; // Number of element init inside (0, 1, 2 if context and max = 2)

    // Size for one element of context
    int sizeContextRavel;

    // Size element context
    int sizeElement;

    // Size of context
    int sizeContext;

    // Methods
    int computeHash(uint8_t* keyArray,
                     int sizeKeyArray);
    /*void ravelArray(uint8_t* arrayRaw,
                    uint8_t* arrayRavel);*/

  public:
    // Constructor
    ContextMapper(int sizeC, 
                  int sizeE);

    // Update use FIFO
    void update(uint8_t* keyArray, // Used to compute hash
                int sizeKeyArray,
                uint8_t* data,
                int sizeData);
    void createKey(int keyValue);
    void updateKey(int keyValue, 
                   uint8_t* data,
                   int sizeData);

    // Others
    void concatArray(uint8_t* arrayA,
                    int sizeArrayA,
                    uint8_t* arrayB,
                    int sizeArrayB,
                    uint8_t* arrayConcat);
       

    // Getter and setter
    void setCumul(float* freqs, 
                  uint64_t* cumul);
                  
    uint8_t* getDataContext(
      uint8_t* keyArray, int sizeKeyArray);

    /*uint8_t* getDataContextRavel(
      uint8_t* keyArray, int sizeKeyArray);*/

    int getContextSize(
      uint8_t* keyArray, int sizeKeyArray);

    int getSizeDataContext();
    int getSizeDataContextRavel();
    int getSizeElement();
    int getSizeContext();
    
};


#endif
