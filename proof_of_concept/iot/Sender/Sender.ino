
// Importations

// // Send data
#include "esp_netif.h"
#include "esp_wifi.h"

// // Receive data
#include <SPI.h>
#include <WiFi.h>

// // Custom protocols
#include "udp_protocol.h"
#include "ipv4_protocol.h"
#include "ipv6_protocol.h"
#include "wifi_protocol.h"

// // Others
#include "wifi_protocol.h"
#include "arithmetic_coder.h"
#include "compressor.h"
#include "context_mapper.h"


/*
 * Define constants
 */


// Define variables
#define TIMESTEPS          16 // Set as attribut ?!
#define SIZE_PACKET_BIN    800 //464 // En bit
#define SIZE_PACKET_HEX    100 //58 // En bit
#define SIZE_CONTEXT       2
#define NB_EQUIPMENT       2

// Activate compression/decompression mode
#define MODE_COMPRESS      true 


/*
 * Define variables
 */


const char* ssid = "dell";

// Protocols instanciations

//CoapProtocol coapProtocol;
UdpProtocol udpProtocol;
Ipv4Protocol ipv4Protocol;
Ipv6Protocol ipv6Protocol;
WifiProtocol wifiProtocol;

// MAC config

uint8_t macSrcAddress[6] = {
  0x50, 0x02, 0x91, 0x9c, 0x88, 0x8c}; //"50:02:91:9c:88:8c";
uint8_t macDstAddress[6] = {
  0x00, 0x42, 0x38, 0x17, 0x89, 0x1e}; //"00:42:38:17:89:1e";

// Data

int sizeData;
uint8_t data[15] = {
  0x54, 0x65, 0x6d, 0x70, // Text is: "Temperature: 00"
  0x65, 0x72, 0x61, 0x74, 
  0x75, 0x72, 0x65, 0x3a, 
  0x20, 0x30, 0x30};

// Ipv4 config

// // Address

IPAddress ip(10, 42, 0, 1); 
IPAddress dns(8, 8, 8, 8);
IPAddress gateway(10, 42, 0, 78);
String ipv4SrcAddress = ip.toString();
String ipv4DstAddress = gateway.toString();
int sizeIpv4SrcAddress = 9;
int sizeIpv4DstAddress = 10;

// // Data

uint8_t *ipv4Data;
int ipv4SizeData;
uint8_t *ipv4DataPadded;
int ipv4SizeDataPadded;

uint8_t *ipv4DataBin;
int ipv4SizeDataBin;
uint8_t *ipv4DataBinPadded;
int ipv4SizeDataBinPadded;

// Ipv6 config

// // Address

uint8_t ipv6DstAddress[16] = { // fe80::507d:c8e7:ac51:826
    0xfe, 0x80, 0x00, 0x00,
    0x50, 0x7d, 0xc8, 0xe7,
    0xac, 0x51, 0x08, 0x26};
uint8_t ipv6SrcAddress[16] = { // fe80::5002:91ff:fe9c:888c
    0xfe, 0x80, 0x00, 0x00,
    0x50, 0x02, 0x91, 0xff,
    0xfe, 0x9c, 0x88, 0x8c};

// // Data

uint8_t* ipv6Data;
int ipv6SizeData;

uint8_t* ipv6DataBin;
int ipv6SizeDataBin;

// UDP config

// // Num port

uint8_t dstPort[2] = {
  0x1f, 0x90}; // 8080
uint8_t srcPort[2] = {
  0x08, 0x00}; // 2048

// // UDP data

uint8_t* udpData;
int udpSizeData;

// Wifi

uint8_t* frame;
int sizeFrame;

// Data compress

uint8_t* dataCompress;
int sizeDataCompress;

uint8_t* dataCompressPadded;
int sizeDataCompressPadded;

// Context Mapper

uint8_t* keyContext;
int sizeKeyContext;
ContextMapper* contextMapper;

// Compressor

Compressor* compressor;


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
  // Define index for bit

  int index_bit = 7;
  
  // For each block of 8 bits

  for (int i=0; i<sizeArrayBin; i=i+8)
  {
    
    // Reduce each block to an integer
    
    for (int j=i; j<i+8; j++)
    {
      arrayHex[i/8] = arrayHex[i/8] + arrayBin[j]*pow(2, index_bit);
      index_bit--;
      
    }

    // Reset index bit

    index_bit = 7;
    
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

  // SUPER IMPORTANT !
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

  setup_connection();

  // Data

  sizeData = sizeof(data)/sizeof(data[0]);

  // Ipv4 config
  
  ipv4Protocol.setSrcAddress(
    ipv4SrcAddress, sizeIpv4SrcAddress, true);
  ipv4Protocol.setDstAddress(
    ipv4DstAddress, sizeIpv4DstAddress, true);
  
  // Ipv6 config
  
  ipv6Protocol.setDstAddress(
    ipv6SrcAddress);
  ipv6Protocol.setDstAddress(
    ipv6SrcAddress);
  
  // UDP config
 
  udpProtocol.setSrcPort(
    srcPort, true); // 8080
  udpProtocol.setDstPort(
    dstPort, true); // 8080
  
  // Context Mapper
  
  sizeKeyContext = 12;
  keyContext = new uint8_t[12];
  concatArray(macSrcAddress, 6,
              macDstAddress, 6,
              keyContext);
  contextMapper = new ContextMapper(
    SIZE_CONTEXT, SIZE_PACKET_BIN);
  
  // Compressor
  
  compressor = new Compressor(
      2, contextMapper, TIMESTEPS,
      SIZE_PACKET_BIN, SIZE_PACKET_HEX,
      SIZE_CONTEXT);

  printf("[DEBUG][setup] setup() finish ! \n");
}


void loop() {

  // Update keyContext if necessary

  // Frame creation

  // // Update data

  // updateData()
  sizeData = sizeof(data)/sizeof(data[0]);
  
  // // Encapsulation UDP(Data)
  
  udpProtocol.setPayload(
    data, sizeData, true, true);

  // // Encapsulation IPv4(UDP(Data))
  
  udpData = udpProtocol.getData();
  udpSizeData = udpProtocol.getSizeData();
  ipv4Protocol.setPayload(
    udpData, udpSizeData, true);

  // Apply padding

  // // Get data
  
  ipv4Data = ipv4Protocol.getData();
  ipv4SizeData = ipv4Protocol.getSizeData();

  // // Create padded array

  ipv4SizeDataPadded = ipv4SizeData + (TIMESTEPS / 8);  // Be careful if the padding is not a modulo of 8 ! 
  ipv4DataPadded = new uint8_t[ipv4SizeDataPadded];

  // // Fill with 0 at the beginning

  for(int i = 0; i < (TIMESTEPS/8); i++) { 
    ipv4DataPadded[i] = 0;
  } 

  // // Fill with the other part

  for(int i = (TIMESTEPS/8); i < ipv4SizeDataPadded; i++) { 
    ipv4DataPadded[i] = ipv4Data[i-(TIMESTEPS/8)];
  } 

  // Header compression

  compressor->compress(ipv4DataPadded,
                       ipv4SizeDataPadded,
                       keyContext,
                       sizeKeyContext);
  dataCompressPadded = compressor->getDataCompressHex();
  sizeDataCompressPadded = compressor->getSizeDataCompressHex();


  // // Remove padding

  sizeDataCompress = sizeDataCompressPadded - (TIMESTEPS / 8);
  dataCompress = new uint8_t[sizeDataCompress];

  for(int i = 0; i < sizeDataCompress; i++) { 
    dataCompress[i] = dataCompressPadded[i + (TIMESTEPS / 8)];
  } 
  
  // // Encapsulation Wifi(IPv4(UDP))
  
  wifiProtocol.setPayload(
    dataCompress, sizeDataCompress);
  Serial.println("WifiProtocol Data  : ");
  wifiProtocol.printData();

  // Frame sending

  // // Array extraction
  
  frame = wifiProtocol.getData();
  sizeFrame = wifiProtocol.getSizeData();

  // SET/UPDATE context

  // // Convert to bin

  ipv4SizeDataBin = ipv4SizeData*8;
  ipv4DataBin = new uint8_t[ipv4SizeDataBin];
  convertHexToBin(ipv4Data,
                  ipv4SizeData,
                  ipv4DataBin);

  // // Update context    

  printf("[DEBUG][loop] UPDATE CONTEXT ! \n");
  contextMapper->update(
    keyContext,
    sizeKeyContext,
    ipv4DataBin,
    ipv4SizeDataBin);
  printf("[DEBUG][loop] CONTEXT UPDATE ! ");

  // // Sending
  
  send_data(frame, sizeFrame);
  Serial.println("\n");
  delay(3000);

  // Free mémory space
  
  delete [] ipv4DataBin;
  delete [] ipv4DataBinPadded;
  
}
