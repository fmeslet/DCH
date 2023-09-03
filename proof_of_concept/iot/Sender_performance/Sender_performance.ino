
// Importations

// // Send data
#include "esp_netif.h"
#include "esp_wifi.h"

// // Receive data
#include <SPI.h>
#include <WiFi.h>

// // Custom protocols
#include "wifi_protocol.h"

// // Others
#include "wifi_protocol.h"
#include "arithmetic_coder.h"
#include "compressor.h"
#include "context_mapper.h"
#include "packets_bit.h"


// Time measurement
// From: https://forum.arduino.cc/t/measuring-time/96602 
unsigned long startTime;
unsigned long currentTime;
unsigned long elapsedTime;


/*
 * Define constants
 */


// Define variables
#define TIMESTEPS          16 // Set as attribut ?!
#define SIZE_PACKET_BIN    320 // En bit
#define SIZE_PACKET_HEX    40 // En bit
#define SIZE_CONTEXT       2

// Activate compression/decompression mode
#define MODE_COMPRESS      true


/*
 * Define variables
 */


const char* ssid = "dell";

// Protocols instanciations

WifiProtocol wifiProtocol;

// MAC config

/*uint8_t macSrcAddress[6] = {
  0x50, 0x02, 0x91, 0x9c, 0x88, 0x8c}; //"50:02:91:9c:88:8c";
uint8_t macDstAddress[6] = {
  0x00, 0x42, 0x38, 0x17, 0x89, 0x1e}; //"00:42:38:17:89:1e";*/

// Config

IPAddress ip(10, 42, 0, 1); 
IPAddress dns(8, 8, 8, 8);
IPAddress gateway(10, 42, 0, 78);
String ipv4SrcAddress = ip.toString();
String ipv4DstAddress = gateway.toString();
int sizeIpv4SrcAddress = 9;
int sizeIpv4DstAddress = 10;

// Wifi

uint8_t* frame;
int sizeFrame;

// Data compress

uint8_t* dataCompressBit;
int sizeDataCompressBit;

uint8_t* dataCompressBitPadded;
int sizeDataCompressBitPadded;

uint8_t* dataCompressHex;
int sizeDataCompressHex;

uint8_t* dataCompressHexPadded;
int sizeDataCompressHexPadded;

// Context Mapper

uint8_t* keyContext;
int sizeKeyContext;
ContextMapper* contextMapper;

// Compressor

Compressor* compressor;


// Informations

// // Change packet to compress

int counterPackets;

// // Packet to compress

uint8_t* packet;
int sizePacket;

uint8_t* packetPadded;
int sizePacketPadded;

// // Context informations

uint8_t* contextPacket;
int sizeContextPacket;
int contextSizePacket;


/*
 * Define functions
 */


esp_err_t esp_wifi_80211_tx(
  wifi_interface_t ifx, const void *buffer, 
  int len, bool en_sys_seq);


void send_data(uint8_t* frame,
               int lengthFrame) {

  Serial.println("Send data !");
  esp_wifi_80211_tx(WIFI_IF_STA, frame, 
                    //sizeof(frame), 
                    lengthFrame,
                    true);
}

void convertHexToBin(
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

void convertBinToHex(
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



void concatArray(uint8_t* arrayA,
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


void setup_connection()
{
  // Get config
  
  Serial.print("[NEW] ESP32 Board MAC Address: ");
  Serial.println(WiFi.macAddress());

  WiFi.config(ip,
              gateway,
              dns);
  
  // Activate Ipv6
  // WiFi.enableIpV6();

  WiFi.mode(WIFI_STA); // Optional
  WiFi.begin(ssid);
  Serial.println("\nConnecting");

  while(WiFi.status() != WL_CONNECTED){
      Serial.print(".");
      delay(90);
  }

  // Wifi promiscuous
  esp_wifi_set_promiscuous(true);

  Serial.println("\nConnected to the WiFi network");
  Serial.print("Local ESP32 IP: ");
  Serial.println(WiFi.localIP());
  //Serial.println(WiFi.localIPv6());
  
  Serial.print("Gateway IP: ");
  Serial.println(WiFi.gatewayIP());
  
}


/*
 * Code start
 */
 

void setup() {
  
  Serial.begin(115200);

  // Setup connection

  //setup_connection();
  
  // Context Mapper
  
  sizeKeyContext = 1;
  keyContext = new uint8_t[sizeKeyContext];
  contextMapper = new ContextMapper(
    SIZE_CONTEXT, SIZE_PACKET_BIN);
      
  // Compressor
  
  compressor = new Compressor(
      2, contextMapper, TIMESTEPS,
      SIZE_PACKET_BIN, SIZE_PACKET_HEX,
      SIZE_CONTEXT);

  //printf("[DEBUG][setup] setup() finish ! \n");

  // Set counter
  
  counterPackets = 0;

}


void loop() {

  // Extract packet and context

  packet = packets[counterPackets];
  sizePacket = sizePackets[counterPackets];
  
  contextPacket = \
    contextPackets[counterPackets];
  sizeContextPacket = \
    sizeContextPackets[counterPackets];
  contextSizePacket = \
    contextSizePackets[counterPackets];

  /*printf("[DEBUG][loop] Take context and print ! \n");

  Serial.print("[ "); 
  for(int i = 0; i < sizeContextPacket-1; i++) { 
    Serial.print(contextPacket[i]);
    Serial.print(", ");
  }

  // Print last element
  Serial.print(contextPacket[sizeContextPacket-1]);
  Serial.println(" ]");

  printf("[DEBUG][loop] sizeContextPacket: %d \n", sizeContextPacket);*/

  // Process context

  keyContext[0] = counterPackets;
  contextMapper->update(
        keyContext,
        sizeKeyContext,
        contextPacket,
        sizeContextPacket,
        contextSizePacket);

  uint8_t* dataContextTest = contextMapper->getDataContext(
    keyContext, 1);

  /*printf("[DEBUG][loop] Print context ! \n");

  Serial.print("[ "); 
  for(int i = 0; i < sizeContextPacket-1; i++) { 
    Serial.print(dataContextTest[i]);
    Serial.print(", ");
  }

  // Print last element
  Serial.print(dataContextTest[sizeContextPacket-1]);
  Serial.println(" ]");*/


/*  printf("[DEBUG][loop] dataCompress :\n");

  Serial.print("[ "); 
  for(int i = 0; i < sizeDataCompress-1; i++) { 
    Serial.print(dataCompress[i]);
    Serial.print(", ");
  }

  // Print last element
  Serial.print(dataCompress[sizeDataCompress-1]);
  Serial.println(" ]");*/
  

  // Apply padding to packet

  // // Create padded array

  sizePacketPadded = ipv4SizeData + (TIMESTEPS / 8);  // Be careful if the padding is not a modulo of 8 ! 
  packetPadded = new uint8_t[sizePacketPadded];

  // // Fill with 0 at the beginning

  for(int i = 0; i < (TIMESTEPS/8); i++) { 
    packetPadded[i] = 0;
  } 

  // // Fill with the other part

  for(int i = (TIMESTEPS/8); i < sizePacketPadded; i++) { 
    packetPadded[i] = packet[i-(TIMESTEPS/8)];
  }   


  // Header compression

  startTime = millis();
  printf("[DEBUG][loop] startTime : %lu \n", startTime);

  if (MODE_COMPRESS)
  {
    compressor->compress(packetPadded,
                         sizePacketPadded,
                         keyContext,
                         sizeKeyContext);
    dataCompressHexPadded = compressor->getDataCompressHex();
    sizeDataCompressHexPadded = compressor->getSizeDataCompressHex();
    sizeDataCompressBitPadded = sizeDataCompressHex*8;

    sizeDataCompressHex = sizeDataCompressHexPadded - (TIMESTEPS / 8);
    dataCompressHex = new uint8_t[sizeDataCompressHex];

  } 
  else 
  {
    sizeDataCompressBit = sizePacket;
    sizeDataCompressHex = ceil(sizePacket / 8.);
    dataCompressHex = new uint8_t[sizeDataCompressHex];
    convertBinToHex(
      packet, sizePacket, dataCompressHex);
  }
  
  currentTime = millis();

  // Print elapsed time
  
  elapsedTime = currentTime - startTime;
  printf("[DEBUG][loop] Compression time : %lu \n", elapsedTime);

  // Compression ratio


  printf("[DEBUG][loop] sizePacket : %d \n", 
      sizePacket);
  printf("[DEBUG][loop] sizeDataCompressBit : %d \n", 
      sizeDataCompressBit);
  printf("[DEBUG][loop] sizeDataCompressHex : %d \n", 
      sizeDataCompressHex);
  printf("[DEBUG][loop] Compression ratio : %f \n", 
      (float)((float)sizePacket / (float)sizeDataCompressBit));


  // // Rmove padding before sending


  for(int i = 0; i < sizeDataCompressHex; i++) { 
    dataCompressHex[i] = dataCompressHexPadded[i + (TIMESTEPS / 8)];
  }  

  // // Encapsulation Wifi(Packet)
  
  wifiProtocol.setPayload(
    dataCompressHex, sizeDataCompressHex);

  // Frame sending

  // // Array extraction
  
  frame = wifiProtocol.getData();
  sizeFrame = wifiProtocol.getSizeData();

  // // Sending
  
  //send_data(frame, sizeFrame);
  Serial.println("\n");
  delay(3000);

  // Add packet counter

  counterPackets = counterPackets + 1;
  if (counterPackets % PACKETS_QUANTITY == 0) {
    
    // Reset counter
    counterPackets = 0;
  }

  // Free mémory space

  if (! MODE_COMPRESS)
  {
    delete[] dataCompressHex;
  }
  
}
