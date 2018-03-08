# ------------------------------------------------------
# Author : Redwanul Karim
# Assumptions : Annotation for only face data
# Annotation Style : YOLO
# API : Face Plus Plus (https://www.faceplusplus.com/)
# ------------------------------------------------------

import os
import requests
import cv2, asyncio
import aiohttp
from   lxml  import etree
from   json  import JSONDecoder
import xml.etree.cElementTree as ET

# Since this repo is private, Sharing my API KEY and SECRET KEY.

API_KEY    = 'y6sSW0jSltyiXBS8W1Ag6kONq5eqoF6a'   
API_SECRET = 'PZ63DCwUeLMH0wtbUilIDMd-uS5PLoME'
API_URL    = 'https://api-us.faceplusplus.com/facepp/v3/detect'

class Annotation:
    
    object  = None
    api_key = None
    secret_key = None
    training_data_path = None

    def __init__(self, api_key, secret_key, object_type, training_data_path):
        self.api_key    = api_key
        self.secret_key = secret_key
        self.training_data_path = training_data_path
        self.object     = object_type

    async def prepareTrainingData( self ):
        # This dictionary will contain the requested data config
        #     
        idx = 0

        for img in os.scandir( self.training_data_path ):
            
            file_name       = img.name.split('.')[0] + '.xml'
            fileToBeChecked = os.path.dirname(os.path.realpath(__file__)) + '/Annotations/' + file_name
            
            if os.path.isfile( fileToBeChecked ):
                continue

            data = {
                'api_key'    : self.api_key,
                'api_secret' : self.secret_key,
                'image_file' : open(img.path, 'rb')
            }

            async with aiohttp.ClientSession() as session:
                async with await session.post(API_URL, data=data) as resp :
            
                    file      = open('AnnotaionDatas.txt', 'a')
                    # dump      = open('dumpAll.txt', 'ab')
                    # print( self.getCoordinate(resp_data) )
                    resp_data = await resp.json()

                    object_details = {
                        'image_name'    : img.name,
                        'image_path'    : img.path,
                        'face_location' : await self.getCoordinate(resp_data)
                    }

                    if object_details['face_location'] == None:
                        os.remove( object_details['image_path'] )  # As this image does not contain any face
                        continue

                    await self.generateAnnotation(img, object_details, 'Annotations')
                    file.write(str(object_details) + "\n")
                    # dump.write(resp.content)   
                    print("Generated Annotations : ", idx+1)
                    print("Annotation completed for image : ", img.name )
                    idx += 1

    async def getCoordinate( self, data ):
        location_list = []

        if 'faces' not in data:
            return None

        for singleData in data['faces']:
            # print("===>")
            # print(singleData['face_rectangle']['left'])
            location_list.append({
                    'left'   : singleData['face_rectangle']['left'],
                    'top'    : singleData['face_rectangle']['top'],
                    'width'  : singleData['face_rectangle']['width'],
                    'height' : singleData['face_rectangle']['height']
                })
        return location_list
    

    # Ref : https://github.com/markjay4k/YOLO-series/blob/master/part7%20-%20generate_xml.py

    async def generateAnnotation(self, img, object_details, savedir):
        if not os.path.isdir(savedir):
            os.mkdir(savedir)

        image = cv2.imread( object_details['image_path'] )
        height, width, depth = image.shape

        annotation = ET.Element('annotation')
        ET.SubElement(annotation, 'folder').text    = self.training_data_path
        ET.SubElement(annotation, 'filename').text  = object_details['image_name']
        ET.SubElement(annotation, 'segmented').text = '0'
        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text  = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text  = str(depth)
        for location in object_details['face_location']:
            ob = ET.SubElement(annotation, 'object')
            ET.SubElement(ob, 'name').text = self.object
            ET.SubElement(ob, 'pose').text = 'Unspecified'
            ET.SubElement(ob, 'truncated').text = '0'
            ET.SubElement(ob, 'difficult').text = '0'
            bbox = ET.SubElement(ob, 'bndbox')
            ET.SubElement(bbox, 'xmin').text = str(location['left'])
            ET.SubElement(bbox, 'ymin').text = str(location['top'])
            ET.SubElement(bbox, 'xmax').text = str(location['left'] + location['width'])
            ET.SubElement(bbox, 'ymax').text = str(location['top'] + location['height'])

        xml_str = ET.tostring(annotation)
        root = etree.fromstring(xml_str)
        xml_str = etree.tostring(root, pretty_print=True)
        save_path = os.path.join(savedir, img.name.replace('jpg', 'xml'))
        with open(save_path, 'wb') as temp_xml:
            temp_xml.write(xml_str)

# If the files are not named in a sorted order
# you can use this to rename them all

def renameAll( ):
    path = os.path.dirname(os.path.realpath(__file__)) + '/images'
    
    files = ( file for file in os.listdir(path) 
            if os.path.isfile(os.path.join(path, file)))
 
    for i, file in enumerate( files ):
        f_name = file.split('.')
        extension = f_name[1]
        os.rename(path + '/' + file, path + '/' + 'batch-02-{:07}.'.format(i) + extension )

def main():
    # renameAll()
    ann  = Annotation(API_KEY, API_SECRET, 'face', 'images')
    loop = asyncio.get_event_loop( )
    loop.run_until_complete(ann.prepareTrainingData( ))
    loop.close( )

if __name__ == '__main__':
    main( )

