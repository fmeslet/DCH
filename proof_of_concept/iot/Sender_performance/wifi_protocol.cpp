#include "wifi_protocol.h"

/*
 * CONSTRUCTOR
 */
                           
WifiProtocol::WifiProtocol():
         _frameControl{0x08, 0x01},
         _duration{0x30, 0x00},
         _dstAddress{0x00, 0x42, 0x38, 0x17, 0x89, 0x1e},
         _srcAddress{0x50, 0x02, 0x91, 0x9c, 0x88, 0x8c},
         _filtAddress{0x00, 0x42, 0x38, 0x17, 0x89, 0x1e}, // Filtering adress (BSSID)
         _sequence{0x00, 0x00},
         _qos{0x00, 0x00},
         _llc{0xaa, 0xaa, 0x03, 0x00, 0x00, 0x00, 0x08, 0x00}
{
  // Set size
  int sizeArray;
  int indexPos = 0; 

  // Frame control
  _header[indexPos] = _frameControl[0];
  _header[indexPos+1] = _frameControl[1];
  indexPos = indexPos + 2;

  // Duration
  _header[indexPos] = _duration[0];
  _header[indexPos+1] = _duration[1];
  indexPos = indexPos + 2;

  // Dst address
  sizeArray = sizeof(_dstAddress)/sizeof(_dstAddress[0]);
  for(int i=indexPos; i<indexPos+sizeArray; i++)
  {
    _header[i] = _dstAddress[i-indexPos];
  }
  indexPos = indexPos + sizeArray;

  // Src address
  sizeArray = sizeof(_dstAddress)/sizeof(_dstAddress[0]);
  for(int i=indexPos; i<indexPos+sizeArray; i++)
  {
    _header[i] = _srcAddress[i-indexPos];
  }
  indexPos = indexPos + sizeArray;

  // Filt address
  sizeArray = sizeof(_srcAddress)/sizeof(_srcAddress[0]);
  for(int i=indexPos; i<indexPos+sizeArray; i++)
  {
    _header[i] = _srcAddress[i-indexPos];
  }
  indexPos = indexPos + sizeArray;

  // Sequence
  _header[indexPos] = _sequence[0];
  _header[indexPos+1] = _sequence[1];
  indexPos = indexPos + 2;

  // QoS
  _header[indexPos] = _qos[0];
  _header[indexPos+1] = _qos[1];
  indexPos = indexPos + 2;

  // LLC
  sizeArray = sizeof(_llc)/sizeof(_llc[0]);
  for(int i=indexPos; i<indexPos+sizeArray; i++)
  {
    _header[i] = _llc[i-indexPos];
  }
  indexPos = indexPos + sizeArray;

  // Define header size
  _sizeHeader = indexPos;
  _sizePayload = 0;

};



/*
 * PRINTER
 */


void WifiProtocol::printHeader()
{
  Serial.print("[ "); 
  for(int i = 0; i < _sizeHeader-1; i++) { 
    Serial.print(_header[i]);
    Serial.print(", ");
  }

  // Print last element
  Serial.print(_header[_sizeHeader-1]);
  Serial.println(" ]");
}

void WifiProtocol::printPayload()
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

void WifiProtocol::printData()
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



uint8_t* WifiProtocol::getData()
{
  int sizeData = this->getSizeData();
  uint8_t* data = new uint8_t[sizeData];
  
  // Set header
  for(int i=0; i<=_sizeHeader; i++)
  {
    data[i] = _header[i];
  }

  // Set payload
  for(int i=_sizeHeader; i<=sizeData; i++)
  {
    data[i] = _payload[i-_sizeHeader];
  }
  
  return data;
}

int WifiProtocol::getSizeData()
{
  int sizeData = _sizeHeader+_sizePayload;
  return sizeData;
}

uint8_t* WifiProtocol::getHeader()
{
  return _header; 
}

int WifiProtocol::getSizeHeader()
{
  return _sizeHeader; 
}

uint8_t* WifiProtocol::getPayload()
{
  return _payload; 
}

int WifiProtocol::getSizePayload()
{
  return _sizePayload; 
}

uint8_t* WifiProtocol::getFrameControl()
{
  return _frameControl; 
}

uint8_t* WifiProtocol::getDuration()
{
  return _duration; 
}

uint8_t* WifiProtocol::getDstAddress()
{
  return _dstAddress;
}


/*
 * SETTER
 */


void WifiProtocol::setHeader(uint8_t *header,
                                int sizeHeader)
{
  memcpy(_header, header, sizeHeader);
  _sizeHeader = sizeHeader;
  //_sizeHeader = sizeof(_header)/sizeof(_header[0]);
}

void WifiProtocol::setPayload(uint8_t *payload,
                                 int sizePayload)
{
  memcpy(_payload, payload, sizePayload);
  _sizePayload = sizePayload;
}

void WifiProtocol::setFrameControl(uint8_t &frameControl)
{
  *_frameControl = frameControl;
}

void WifiProtocol::setDuration(uint8_t &duration)
{
  *_duration = duration;
}

void WifiProtocol::setDstAddress(uint8_t &dstAddress)
{
  *_dstAddress = dstAddress;
}
