# ------------------------------------------------------
# Assumptions : Annotation for only face data
# Annotation Style : YOLO
# API : face_recognition  
# NOTE: It's mendatory to install face_recognition api ( pip3 install face_recognition )
# ------------------------------------------------------

import os, cv2
import requests, sys
from   PIL   import Image
import face_recognition
from   lxml  import etree
from   json  import JSONDecoder
import xml.etree.cElementTree as ET

class Annotation:
    object  = None
    training_data_path = None
    annotation_path    = None

    def __init__(self, object_type, training_data_path, annotation_path):
        self.training_data_path = training_data_path
        self.object     = object_type
        self.annotation_path = annotation_path

    def prepareTrainingData( self ):

        idx = 0

        for img in os.scandir( self.training_data_path ):
            
            file_name       = img.name.split('.')[0] + '.xml'
            fileToBeChecked = os.path.dirname(
                              os.path.realpath(__file__) ) + '/' + self.annotation_path + '/' + file_name
            if os.path.isfile( fileToBeChecked ):
                continue
        
            file      = open('AnnotaionDatas.txt', 'a')
            # dump      = open('dumpAll.txt', 'ab')
            # print( self.getCoordinate(resp_data) )

            path = os.path.dirname(os.path.realpath(__file__))
            path = str( os.path.join(path, img) )
            # print("Path1 ---> ", path)
            # print("Path2 ---> ", img.path)
            image     = face_recognition.load_image_file( path ) # reading image as a numpy array

            print("Processing :: " + str(img))
            # print("Array Formate :: "  + str(image))

            resp_data = face_recognition.face_locations ( image, number_of_times_to_upsample=0, model="cnn" )
            
            object_details = {
                'image_name'    : img.name,
                'image_path'    : img.path,
                'face_location' : self.getCoordinate(resp_data)
            }

            # for obj in object_details['face_location']:
            #     top, right, bottom, left = obj['top'], obj['right'], obj['bottom'], obj['left']
            #     print("A face is located at pixel location Top: {}, Left: {}, 
            #            Bottom: {}, Right: {}".format(top, left, bottom, right))
            #     face_image   = image[ top:bottom, left:right ]
            #     cropped_face = Image.fromarray(face_image)
            #     cropped_face.show( )

            if object_details['face_location'] == None:
                os.remove( object_details['image_path'] )  # As this image does not contain any face
                continue

            self.generateAnnotation(img, object_details, self.annotation_path)
            file.write(str(object_details) + "\n")
            # dump.write(resp.content)   
            print("Generated Annotations : ", idx+1)
            print("Annotation completed for image : ", img.name )
            idx += 1

    def getCoordinate( self, data ):
        location_list = []

        if data is None:
            return None

        for singleData in data:
            # print("===>")
            # print(singleData['face_rectangle']['left'])
            top, right, bottom, left = singleData
            location_list.append({
                    'left'   : left,
                    'top'    : top,
                    'right'  : right,
                    'bottom' : bottom
            })
        return location_list
    
    # Ref : https://github.com/markjay4k/YOLO-series/blob/master/part7%20-%20generate_xml.py
    
    def generateAnnotation(self, img, object_details, savedir):
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
            ET.SubElement(bbox, 'xmax').text = str(location['right'])
            ET.SubElement(bbox, 'ymax').text = str(location['bottom'])

        xml_str = ET.tostring(annotation)
        root = etree.fromstring(xml_str)
        xml_str = etree.tostring(root, pretty_print=True)
        save_path = os.path.join(savedir, img.name.replace('jpg', 'xml'))
        with open(save_path, 'wb') as temp_xml:
            temp_xml.write(xml_str)


def main( arguments ):

    object_type, dataset_path, annotation_path = arguments[ 1 ], arguments[ 2 ], arguments[ 3 ]
    ann  = Annotation( object_type, dataset_path, annotation_path )
    ann.prepareTrainingData( )

if __name__ == '__main__':
    if(len( sys.argv ) == 4 ):
        main( sys.argv )
    else:
        print("Error:: Please provide object type, dataset location, and annotation destination!\n")
        print("Command::\n~ pythonAnnotation_with_local_api.py object_type dataset_path annotation_path\n")
