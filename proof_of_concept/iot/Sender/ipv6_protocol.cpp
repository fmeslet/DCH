#include "ipv6_protocol.h"


/*
 * CONSTRUCTOR
 */

                           
Ipv6Protocol::Ipv6Protocol():
         _versionTrafficClass(0x60),
         _trafficClassFlowLabel{0x00, 0x00, 0x00},
         _payloadLength{0x00, 0x00}, // Header + payload to compute ?
         _nextHeader{0x02, 0x12},
         _hopLimit(0xff),
         _srcAddress{0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00},
         _dstAddress{0xff, 0xff, 0xff, 0xff,
                     0xff, 0xff, 0xff, 0xff,
                     0xff, 0xff, 0xff, 0xff,
                     0xff, 0xff, 0xff, 0xff}
{

  // Version Protocol and Traffic Class
  _header[IPv6_VERSION_TC_P] = _versionTrafficClass;

  // Traffic Class end Flow Label
  _header[IPv6_TC_FL_P] = _trafficClassFlowLabel[0];
  _header[IPv6_TC_FL_P+1] = _trafficClassFlowLabel[1];
  _header[IPv6_TC_FL_P+2] = _trafficClassFlowLabel[2];

  // Payload Length
  _header[IPv6_PAYLOAD_LEN_P] = _payloadLength[0];
  _header[IPv6_PAYLOAD_LEN_P+1] = _payloadLength[1];

  // Next Header
  _header[IPv6_NEXT_HEADER_P] = _nextHeader[0];
  _header[IPv6_NEXT_HEADER_P+1] = _nextHeader[1];

  // Src address
  for(int i=IPv6_SRC_P; i<IPv6_SRC_P+16; i++)
  {
    _header[i] = _srcAddress[i-IPv6_SRC_P];
  }

  // Dst address
  for(int i=IPv6_DST_P; i<IPv6_DST_P+16; i++)
  {
    _header[i] = _dstAddress[i-IPv6_DST_P];
  }

  // Define header size
  _sizePayload = 0;

};



/*
 * OTHERS
 */


// FROM : https://ben.akrin.com/?p=1347
void Ipv6Protocol::computeAddressEui64(
  uint8_t* eui64Address, 
  uint8_t* macAddress)
{
  int idxEui64 = 0;

  // Compute Eui64
  for (int i=0; i<6; i++)
  {
    if (i == 3)
    {
      eui64Address[3] = 0xff;
      eui64Address[4] = 0xfe;
      idxEui64 = 5;
    }
    eui64Address[idxEui64] = macAddress[i];
    idxEui64++;
  }
  eui64Address[0] = eui64Address[0]^0x02;
}


void Ipv6Protocol::computeAddressLocalLink(
  uint8_t* ipv6Address, 
  uint8_t* macAddress)
{
  ipv6Address[0] = 0xfe;
  ipv6Address[0] = 0x80;

  uint8_t* eui64Address = new uint8_t[8];
  this->computeAddressEui64(
    eui64Address, macAddress);

  // Set value to Ipv6 address
  for (int i=0; i<16; i++)
  {
    if (i>=7)
    {
      ipv6Address[i] = eui64Address[i-7];
    }
  }
}

// See : <bitset> for memory improvement
uint8_t* Ipv6Protocol::convertBinToHex(
                    uint8_t* arrayBin,
                    int sizeArrayBin)
{
  int sizeArrayHex = (int)sizeArrayBin / 8;
  uint8_t* arrayHex = new uint8_t[sizeArrayHex];
  
  int value = 0;
  int index = 0;

  for (int i=0; i<sizeArrayBin; i=i+8)
  {
    for (int j=0; j<8; j++)
    {
      value = value + pow(2, 7-j)*arrayBin[i+j];
    }
    arrayHex[index] = value;
    index++;
    value = 0;
  }
  return arrayHex;
}

uint8_t* Ipv6Protocol::convertIntToBin(
            uint16_t value)
{
  int sizeValue = sizeof(value)*8;
  uint8_t* arrayBin = new uint8_t[sizeValue];

  // Assume it's a 32 bit int
  for (int i=0; i<sizeValue; i++)
  {
    arrayBin[sizeValue-1-i] = (value >> i) & 1;
  }
  return arrayBin;
}

void Ipv6Protocol::resetPayloadLength()
{
  uint16_t value = this->getSizePayload();
  int sizeValueBin = sizeof(value)*8;
  uint8_t* valueBin = this->convertIntToBin(value);
  uint8_t* valueHex = this->convertBinToHex(
    valueBin, sizeValueBin);

  _payloadLength[0] = valueHex[0];
  _payloadLength[1] = valueHex[1];
  _header[IPv6_PAYLOAD_LEN_P] = _payloadLength[0];
  _header[IPv6_PAYLOAD_LEN_P+1] = _payloadLength[1];

  delete [] valueBin;
  delete [] valueHex;
}


/*
 * PRINTER
 */


void Ipv6Protocol::printHeader()
{
  Serial.print("[ "); 
  for(int i = 0; i < IPv6_HEADER_LEN_V-1; i++) { 
    Serial.print(_header[i]);
    Serial.print(", ");
  }

  // Print last element
  Serial.print(_header[IPv6_HEADER_LEN_V-1]);
  Serial.println(" ]");
}

void Ipv6Protocol::printPayload()
{
  Serial.print("[ "); 
  for(int i = 0; i < _sizePayload-1; i++) { 
    Serial.print(_payload[i]);
    Serial.print(", ");
  }

  // Print last element
  Serial.print(_payload[_sizePayload-1]);
  Serial.println(" ]");
}

void Ipv6Protocol::printData()
{
  uint8_t* data = this->getData();
  int sizeData = this->getSizeData();
  Serial.print("[ "); 
  for(int i = 0; i < sizeData-1; i++) { 
    Serial.print(data[i]);
    Serial.print(", ");
  }

  // Print last element
  Serial.print(data[sizeData-1]);
  Serial.println(" ]");
}


/*
 * GETTER
 */
 

uint8_t* Ipv6Protocol::getData()
{
  int sizeData = this->getSizeData();
  uint8_t* data = new uint8_t[sizeData];
  
  // Set header
  for(int i=0; i<IPv6_HEADER_LEN_V; i++)
  {
    data[i] = _header[i];
  }

  // Set payload
  for(int i=IPv6_HEADER_LEN_V; i<=sizeData; i++)
  {
    data[i] = _payload[i-IPv6_HEADER_LEN_V];
  }
  
  return &data[0];
}

uint8_t Ipv6Protocol::getVersionTrafficClass()
{
  return _versionTrafficClass;
}

uint8_t* Ipv6Protocol::getTrafficClassFlowLabel() 
{
  return _trafficClassFlowLabel; 
}

uint8_t* Ipv6Protocol::getPayloadLength() 
{
  return _payloadLength; 
}



// ...



/*
 * SETTER
 */
    

void Ipv6Protocol::setHeader(uint8_t *header)
{
  memcpy(_header, header, IPv6_HEADER_LEN_V);
}

void Ipv6Protocol::setPayload(uint8_t *payload,
                              int sizePayload,
                              bool resetPayloadLength=true)
{
  _sizePayload = sizePayload;
  if (resetPayloadLength)
  {
    this->resetPayloadLength();
  }
  memcpy(_payload, payload, _sizePayload);
}

void Ipv6Protocol::setVersionTrafficClass(
  uint8_t versionTrafficClass)
{
  _versionTrafficClass = versionTrafficClass;
  _header[IPv6_VERSION_TC_P] = _versionTrafficClass;
}

void Ipv6Protocol::setTrafficClassFlowLabel(
  uint8_t* trafficClassFlowLabel)
{
  memcpy(_trafficClassFlowLabel, trafficClassFlowLabel, 3);
  _header[IPv6_TC_FL_P] = _trafficClassFlowLabel[0];
  _header[IPv6_TC_FL_P+1] = _trafficClassFlowLabel[1];
  _header[IPv6_TC_FL_P+2] = _trafficClassFlowLabel[2];
}

void Ipv6Protocol::setPayloadLength(
  uint8_t* payloadLength)
                                  
{
  memcpy(_payloadLength, payloadLength, 2);
  _header[IPv6_PAYLOAD_LEN_P] = _payloadLength[0];
  _header[IPv6_PAYLOAD_LEN_P+1] = _payloadLength[1];
}

void Ipv6Protocol::setNextHeader(
  uint8_t* nextHeader)
{
  memcpy(_nextHeader, nextHeader, 2);
  _header[IPv6_NEXT_HEADER_P] = _nextHeader[0];
  _header[IPv6_NEXT_HEADER_P+1] = _nextHeader[1];
}

void Ipv6Protocol::setSrcAddressLocalLink(
  uint8_t* macAddress)
{
  uint8_t srcAddress[16];
  this->computeAddressLocalLink(
    srcAddress, macAddress);  
  this->setSrcAddress(srcAddress);
}

void Ipv6Protocol::setSrcAddress(
  uint8_t* srcAddress)
{  
  memcpy(_srcAddress, srcAddress, 16);
  for(int i=IPv6_SRC_P; i<IPv6_SRC_P+16; i++)
  {
    _header[i] = _srcAddress[i-IPv6_SRC_P];
  }
}

void Ipv6Protocol::setDstAddressLocalLink(
  uint8_t* macAddress)
{
  uint8_t dstAddress[16];
  this->computeAddressLocalLink(
    dstAddress, macAddress);  
  this->setDstAddress(dstAddress);
}

void Ipv6Protocol::setDstAddress(
  uint8_t* dstAddress)
{
  memcpy(_dstAddress, dstAddress, 16);
  for(int i=IPv6_DST_P; i<IPv6_DST_P+16; i++)
  {
    _header[i] = _dstAddress[i-IPv6_DST_P];
  }
}
