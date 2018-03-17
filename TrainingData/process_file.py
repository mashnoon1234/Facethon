
# Script for remaing all the files and copying them in a single folder at once.
# Assuming Data Set is FDDB Face Dataset and this script in the root folder of that dataset.

import os, shutil, sys

def FileProcess( batch_no, destination, action ):
    idx   = 0    
    path1 = os.path.dirname(os.path.realpath(__file__))

    # Silly nested for loops :v 

    for x in os.listdir(path1):
        idx += 1
        if not os.path.isfile(path1 + '/' + x):
            for y in os.listdir(path1 + '/' + x):
                idx += 1
                if not os.path.isfile(path1 + '/' + x + '/' + y):
                    for z in os.listdir(path1 + '/' + x + '/' + y):
                        idx += 1
                        if not os.path.isfile(path1 + '/' + x + '/' + y + '/' + z):
                            path = os.path.dirname(path1 + '/' + x + '/' + y + '/' + z) + '/big'
                            files = ( file for file in os.listdir(path) 
                                      if os.path.isfile(os.path.join(path, file)))

                            for i, file in enumerate( files ):
                                f_name = file.split('.')
                                extension = f_name[1]

                                if action == 'rename':
                                    os.rename(path + '/' + file, path + '/' + "batch-" + batch_no + '-' + str(idx) + '-' + '{:07}.'.format(i) + extension )
                                else:  
                                    if( os.path.isfile( os.path.join(path, file) ) ):
                                        print("---> " + str(file))
                                        shutil.copy(str( os.path.join(path, file) ), destination )


def main( arguments ):

    batch_no, destination = arguments[ 1 ], arguments[ 2 ]

    FileProcess( batch_no, destination, action = 'rename')
    FileProcess( batch_no, destination, action = 'copy'  )

if __name__ == '__main__':
    if len(sys.argv ) == 3 :
        main( sys.argv )
    else:
        print("Invalid number of arguments")
        print("Command ::\n~ python process_file.py batch_no destination_path")
