#include "context_mapper.h"


/*
 * CONSTRUCTOR
 */


ContextMapper::ContextMapper(
  int sizeC, int sizeE):
  sizeContext(sizeC),
  sizeElement(sizeE)
{  
  sizeContextRavel = sizeContext*sizeElement;
};


/*
 * OTHERS
 */


// For more information: https://stackoverflow.com/questions/8317508/hash-function-for-a-string
// https://www.partow.net/programming/hashfunctions/
// https://stackoverflow.com/questions/6136964/rs-hashing-program
//  RS Hash Function
int ContextMapper::computeHash(uint8_t* keyArray,
                               int sizeKeyArray)
{
   unsigned int b    = 378551;
   unsigned int a    = 63689;
   unsigned int hash = 0;
   unsigned int i    = 0;

   for (i = 0; i < sizeKeyArray; ++keyArray, ++i)
   {
      hash = hash * a + (*keyArray);
      a    = a * b;
   }

   return hash;
}


void ContextMapper::concatArray(
            uint8_t* arrayA,
            int sizeArrayA,
            uint8_t* arrayB,
            int sizeArrayB,
            uint8_t* arrayConcat)
{
  
  int sizeArrayConcat = sizeArrayA + sizeArrayB;
  
  // For on array A
  for (int i=0; i<sizeArrayA; i++)
  {
    arrayConcat[i] = arrayA[i];
  }

  // For on array B
  for (int i=sizeArrayA; i<sizeArrayConcat; i++)
  {
    arrayConcat[i] = arrayB[i-sizeArrayA];
  }
}



/*
 * Examples:
 * [0, 1, 2, 3, 4, ...]
 * The last one is the 4. The shift produce:
 * [1, 2, 3, 4, 5, ...]
 */

void ContextMapper::updateKey(
  int keyValue, 
  uint8_t* data,
  int sizeData)
{
  // Extract array with key
  uint8_t* contextData = dataContext[
        keyValue];

  // Shift context data
  for (int i=0; i<sizeContext-1; i++) { 
    for (int j=0; j<sizeElement; j++) { 
      contextData[(i*sizeElement) + j] = contextData[(i+1)*sizeElement+j]; 
    }
  }

  // Update last values of context
  for (int j=0; j<sizeData; j++) { 
      contextData[(sizeContext-1)*sizeElement + j] = data[j]; 
  }

  // Fill the other part with 0
  for (int j=sizeData; j<sizeElement; j++) { 
      contextData[(sizeContext-1)*sizeElement+j] = 0; 
  }

  // Add to map
  dataContext[keyValue] = contextData;
  contextSize[keyValue] = min(
    contextSize[keyValue]+1, sizeContext);
  
}


void ContextMapper::update(uint8_t* keyArray,
                      int sizeKeyArray,
                      uint8_t* data,
                      int sizeData) // Check if array is well sized
{
  // Compute key
  int keyValue = this->computeHash(
    keyArray, sizeKeyArray);
  printf("[DEBUG][ContextMapper::update] keyValue : %d \n", keyValue);
  
  // Check if the key exist
  if (dataContext.find(
    keyValue) != dataContext.end())
  {
    this->updateKey(
      keyValue, data, sizeData);
  }
  else
  {
    printf("[DEBUG][ContextMapper::update] keyValue NOT exist \n");
    this->createKey(keyValue);
    this->updateKey( // TO REMOVE ?
      keyValue, data, sizeData);
  }
  
}


void ContextMapper::createKey(
  int keyValue)
{
  // Init context to zero
  uint8_t* contextData = new uint8_t[
        sizeContext*sizeElement];
        
  for (int i=0; i<sizeContext*sizeElement; i++) { 
      contextData[i] = 0; 
  }

  // Add to map
  dataContext[keyValue] = contextData;
  contextSize[keyValue] = 0;

}



/*
 * GETTER
 */



uint8_t* ContextMapper::getDataContext(
  uint8_t* keyArray, int sizeKeyArray)
{
  // Compute key
  int keyValue = this->computeHash(
    keyArray, sizeKeyArray);

  // Check if the key exist
  if (dataContext.find(
    keyValue) != dataContext.end())
  {
    return dataContext[keyValue];
  }
  else
  {
    this->createKey(keyValue);
    return dataContext[keyValue];
  }
}


int ContextMapper::getContextSize(
      uint8_t* keyArray, 
      int sizeKeyArray)
{
  // Compute key
  int keyValue = this->computeHash(
    keyArray, sizeKeyArray);

  // Check if the key exist
  if (contextSize.find(
    keyValue) != contextSize.end())
  {
    contextSize[keyValue]; // return 0;
  }
  else
  {
    contextSize[keyValue] = 0;
  }

  return contextSize[keyValue];
}


int ContextMapper::getSizeElement()
{
  return sizeElement;
}


int ContextMapper::getSizeContext()
{
  return sizeContext;
}
