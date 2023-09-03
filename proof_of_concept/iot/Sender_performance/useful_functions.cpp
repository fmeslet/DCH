#include "useful_functions.h"


void printArray(uint8_t* myArray,
                int sizeArray)
{
  Serial.print("[ "); 
  for(int i = 0; i < sizeArray-1; i++) { 
    Serial.print(myArray[i]);
    Serial.print(", ");
  }

  // Print last element
  Serial.print(myArray[sizeArray-1]);
  Serial.println(" ]");
}

void printArray(char* myArray,
                int sizeArray)
{
  Serial.print("[ "); 
  for(int i = 0; i < sizeArray-1; i++) { 
    Serial.print(myArray[i]);
    Serial.print(", ");
  }

  // Print last element
  Serial.print(myArray[sizeArray-1]);
  Serial.println(" ]");
}


uint8_t* ipAddressStringToHex(String ipAddressString, 
                             int sizeIpAddressArray,
                             char separator)
{
  char ipAddressArray[16]; // Ipv6 length by default
  uint8_t* ipAddressHexArray = new uint8_t[4];

  // Convert string to array char
  strcpy(ipAddressArray, ipAddressString.c_str());

  int counter = 0;
  int idxArray = 0;
  for (int i=0; i<=sizeIpAddressArray; i++) 
  {
    char valA = ipAddressArray[i];
    char valB = '.';
    if (!strncmp(&valB, &valA, 1) || (i == sizeIpAddressArray))
    {
      String s = "";
      for (int j=0; j<counter; j++)
      {
        // Create String
        s = s + ipAddressArray[(i-counter)+j];
      }
      
      // Convert to hexa
      ipAddressHexArray[idxArray] = atoi(s.c_str());
      
      counter = 0;
      idxArray++;
    } else {
      counter++;
    }
  }
  return ipAddressHexArray; 
}


// Or return a bool ?
uint8_t* convertHexaToBit(uint8_t* arrayHex,
                         int sizeArrayHex)
{
  uint8_t* arrayBit = new uint8_t[sizeArrayHex*8];
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
        arrayBit[(i+index)+j] = 1; 
      }
      else
      {
        arrayBit[(i+index)+j] = 0; 
      }
    }
    index = index + 7;
  }
  return arrayBit;
}


// 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,       // 4-9: Destination address (broadcast)

/*
 * uint8_t frame_raw[] = { // 0x80 (subtype = 8) (type=0), 0x00
  0x08, 0x01,             // 0-1: Frame Control
  0x30, 0x00,             // 2-3: Duration
  0x00, 0x42, 0x38, 0x17, 0x89, 0x1e,       // 4-9: Destination address
  0x50, 0x02, 0x91, 0x9c, 0x88, 0x8c,       // 10-15: Source address
  0x00, 0x42, 0x38, 0x17, 0x89, 0x1e,       // 16-21: BSSID
  0x00, 0x00,            // 22-23: Sequence / fragment number
  0x00, 0x00,            // 22-23: QoS control / fragment number
  0xaa, 0xaa, 0x03, 0x00, 0x00, 0x00, 0x08, 0x00,     // 24-31: LLC Layer
  
  // IP
  0x45, // Version and Header Length
  0x00, // Differentiated Services Field
  0x00, 0x28, // Total length : 40
  0x02, 0x12, // Identification 
  0x00, // Flags
  0x00, // Fragment offset
  0xff, // Time to live 
  0x11, // UDP Protocol
  0xed, 0x7e, // Header Checksum
  0x0a, 0x2a, 0x00, 0x4e, // Source adress
  0xc0, 0xa8, 0x01, 0x14, // Destination address
  
  // UDP
  0xc7, 0x1b, // Source port
  0x09, 0x56, // Destination port
  0x00, 0x14, // Length
  0xf0, 0xa5, // Checksum

  // DATA
  0x61, 0x63, 0x6b, 0x6e, 0x6f, 0x77, 
  0x6c, 0x65, 0x64, 0x67, 0x65, 0x64,   
};
 */


  /*char ipAddressArray[20];
  uint8_t ipAddressHexArray[4];
  strcpy(ipAddressArray, ipAddress.c_str());
  int sizeIpAddressArray = 8+3; //sizeof(ipAddress)/sizeof(ipAddress[0]);

  printf("[DEBUG] ipAddressArray : ");
  printArray(ipAddressArray, sizeIpAddressArray);

  printf("[DEBUG] sizeIpAddressArray : %d \n", sizeIpAddressArray);

  int counter = 0;
  int idxArray = 0;
  for (int i=0; i<=sizeIpAddressArray; i++) 
  {
    Serial.println("[DEBUG] sizeIpAddressArray-1 : ");
    Serial.println(sizeIpAddressArray-1);
  
    Serial.println("[DEBUG] i : ");
    Serial.println(i);
    char valA = ipAddressArray[i];
    char valB = '.';
    if (!strncmp(&valB, &valA, 1) || (i == sizeIpAddressArray))
    {
      printf("[DEBUG][if] i value : %d \n", i);
      printf("[DEBUG][if] counter : %d \n", counter);
      
      String s = "";
      for (int j=0; j<counter; j++)
      {
        // Create String
        s = s + ipAddressArray[(i-counter)+j];
      }

      Serial.println("[DEBUG] s : "+s);
      // Convert to hexa
      ipAddressHexArray[idxArray] = atoi(s.c_str());
      
      counter = 0;
      idxArray++;
    } else {
      printf("[DEBUG][else] i value : %d \n", i);
      counter++;
    }
  }*/


// COMPUTE IPv6 ADDRESS


/*
  Serial.print("[NEW] ESP32 Board MAC Address:  ");

  int sizeArrayMac = 6;
  Serial.print("[ "); 
  for(int i = 0; i < sizeArrayMac-1; i++) { 
    Serial.print(macSrcAddress[i]);
    Serial.print(", ");
  }

  // Print last element
  Serial.print(macSrcAddress[sizeArrayMac-1]);
  Serial.println(" ]");

  //Serial.println("Local IP is : "+WiFi.gatewayIP().toString());
  //Serial.println("Gateway IP is : "+WiFi.localIP().toString());

  uint8_t* eui64 = new uint8_t[8];
  uint8_t ipv6SrcAddress[16] = {
    0xfe, 0x80, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00};
  int idxEui64 = 0;

  // Compute Eui64
  for (int i=0; i<6; i++)
  {

    if (i == 3)
    {
      eui64[3] = 0xff;
      eui64[4] = 0xfe;
      idxEui64 = 5;
    }
    eui64[idxEui64] = macSrcAddress[i];
    idxEui64++;
  }

  eui64[0] = eui64[0]^0x02;
  Serial.println("Bit change eui64");
  Serial.println(eui64[0]);

  // Set value to Ipv6 address
  for (int i=0; i<16; i++)
  {
    if (i>=7)
    {
      ipv6SrcAddress[i] = eui64[i-7];
    }
  }

  Serial.println("IPv6 address  : ");

  int sizeArray = 16;
  Serial.print("[ "); 
  for(int i = 0; i < sizeArray-1; i++) { 
    Serial.print(ipv6SrcAddress[i]);
    Serial.print(", ");
  }

  // Print last element
  Serial.print(ipv6SrcAddress[sizeArray-1]);
  Serial.println(" ]");
*/
