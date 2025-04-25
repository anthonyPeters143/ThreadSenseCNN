####################################################################################################
# SCRAPING METHODS
####################################################################################################

# Will convert aand save passed file DST file as a PNG image at the passed image path, will create a temp SVG file used for conversion, will return flag based on valid execution
def convert_pattern_into_png(file_path, image_path):
    # Import libaries
    import pyembroidery
    import cairosvg
    import tempfile
    import xml.etree.ElementTree as ET

    # Attempt conversion
    try:
        # Create temp SVG file for converstion
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as svg_temp_file:
            # Store temp file path
            svg_temp_file_path = svg_temp_file.name

            # Convert DST pattern into SVG using temp file
            pyembroidery.convert(file_path, svg_temp_file_path)

            # Open SVG tree parser
            with open(svg_temp_file_path, 'r') as svg_file:
                tree = ET.parse(svg_file)

            # Loop thorugh every path element in SVG
            for path in tree.getroot().findall(".//svg:path", namespaces={"svg": "http://www.w3.org/2000/svg"}):
                path.set("stroke-width", "5")           # Update stroke width to be wider
                path.set("stroke-linecap", "butt")      # Update stroke to have rounded caps
                path.set("stroke-linejoin", "bevel")    # Update stoke to have flat corners

            # Output updated SVG 
            tree.write(svg_temp_file_path)

            # Convert updated SVG into PNG with backround
            cairosvg.svg2png(url=svg_temp_file_path, write_to=image_path, background_color="white")

            # Set return flag to true
            execute_valid_flag = True
    
    # Conversion failed
    except:
        # Set execute valid flag to false
        execute_valid_flag = False

    # After execution return execute valid flag
    return execute_valid_flag

# Will convert and sac\ve passed file into a PNG file at the passed image path, will return flag based on valid execution
def convert_image_into_png(file_path, image_path):
    # Import libaries
    from PIL import Image

    # Attmept conversion
    try:
        # Open image from file path
        image = Image.open(file_path)

        # Save image at image path
        image.save(image_path, 'PNG')
        
        # Set return flag to true
        execute_valid_flag = True
    
    # Converison falied
    except:
        # Set execute valid to false
        execute_valid_flag = False

    # After execution return execute valid flag
    return execute_valid_flag

# Will create and return a property list of thread count, width (centimeter), height (centimeter), and image path from passed DST file
def determine_entry_data(file_path, image_path):
    # Import libaries
    import pyembroidery

    # Convert DST file into pattern
    pattern = pyembroidery.read_dst(file_path)

    # Check if X, Y, and ST counts are valid numbers
    try:
        if (pattern.extras['+X'].isdigit() and pattern.extras['-X'].isdigit() and 
            pattern.extras['+Y'].isdigit() and pattern.extras['-Y'].isdigit() and
            pattern.extras['ST'].isdigit()):
            
            # Valid
            # Calculate and store width and height from postive and negative X and Y values
            width = float(pattern.extras['+X']) + float(pattern.extras['-X'])
            height = float(pattern.extras['+Y']) + float(pattern.extras['-Y'])

            # Store thread count
            thread_count = pattern.extras['ST']

            # Create return list of properties
            property_list = [image_path, thread_count, width, height]

        else:
            # Invalid
            property_list = []

    # File has incorrect header infomation
    except KeyError:
        property_list = []

    # Return propety list of file
    return property_list
    
# Will prompt user to input thread count, width (centimeter), height (centimeter) for image path
def prompt_user_entry_data(file_path, image_path):
    # Iniltze valid flag to false
    input_valid = False
    
    # Output file path for data
    print("Input data for " + file_path)
    
    # Loop while input is invalid
    while not input_valid:
        try:
            # Attempt to convert user inputs into float 
            width = float(input("\nDesign width (centimeters) : "))
            height = float(input("\nDesign height (centimeters) : "))
            thread_count = float(input("\nDesign thread count : "))

            # Check if the input values are greater than 0
            if width and height and thread_count > 0:
                # Set valid flag to true
                input_valid = True

                # Create return list of properites
                property_list = [image_path, thread_count, width, height]
            
            else:
                # Print invalid error message
                print("ERROR - input is invalid amount, input should be > 0") 


        except:
            # Print input error message
            print("ERROR - input is invalid format, input should be a float value")

    # Return property list of file
    return property_list

# Will update passed CSV writer and image storage using the passed file and valid counter, will create and store a data entry and PNG image from the passed file, if in prediction mode then will prompt for image input, will output progress messages and return values for valid, skipped
def update_dataset_csv(csv_writer, image_storage, file_path, valid_file_counter_ref):
    # Set counter from counter reference
    valid_counter = valid_file_counter_ref[0]

    # Determine if file is a DST for later operations
    dst_input_mode = file_path.lower().endswith('.dst')

    # Create PNG image path
    image_path = image_storage + "\\" + str(valid_counter) + ".png"

    # Get entry data from user input or convert file into a pattern to determine determine
    entry_data = determine_entry_data(file_path, image_path) if dst_input_mode else prompt_user_entry_data(file_path, image_path)

    # Check first the return entry list is not empty and then the image is converted and store at image page within database_images
    if entry_data != [] and convert_pattern_into_png(file_path, image_path) if dst_input_mode else convert_image_into_png(file_path, image_path):
        # Valid
        # Set return values to valid +1 and skipped +0
        outcome_values = 1,0

        # Write entry data to csv file
        csv_writer.writerow(entry_data)

        # Update output message with valid entry data
        output_message = entry_data
    
    else:
        # Invalid
        # Set return values to valid +0 and skipped +1
        outcome_values = 0,1

        # Update output message with invalid entry data
        output_message = f"{file_path} is invalid entry" 

    # Output progress message
    print(output_message)

    # Return outcome values in format valid, skipped
    return outcome_values

# Will recursivly scans through all directories within database path and update dataset using the file infomation, will return flag based on valid execution
def scan_files(csv_writer, image_storage, path=".", valid_file_counter_ref=[0]):
    # Import libaries
    import os

    # Set outcome values to 0
    valid_files = 0
    skipped_files = 0
    total_files = 0

    # Cycle through directory for each path
    with os.scandir(path) as entries:
        for entry in entries:
            # Verfiy file is a DST, PNG, JPEG, or JPG file
            if entry.is_file() and entry.path.lower().endswith((".dst", ".png", ".jpg", ".jpeg")):
                # Attempt to udpate CSV file using passed path and store returned values valid, skipped
                valid, skipped = update_dataset_csv(csv_writer, image_storage, entry.path, valid_file_counter_ref)

                # Update file stats
                valid_files += valid
                skipped_files += skipped
                total_files += 1

            # Recursivly call self
            elif entry.is_dir():
                # Update file stats
                valid, skipped, total = scan_files(csv_writer, image_storage, entry.path, valid_file_counter_ref)
                valid_files += valid
                skipped_files += skipped
                total_files += total

    # Return execute valid flag
    return valid_files, skipped_files, total_files

# Will create CSV file and store at the passed CSV path from the infomation scraped from the passed database path, will remove the old csv file if found and return flag based on valid execution 
def scrape_data_from(csv_path, database_path, image_storage):
    # Import libaries
    import os
    from csv import writer

    # Import constants
    import constants

    try:
        # Remove old csv file
        os.remove(csv_path)

        # Set remove message
        fileRemoveMessage = "Removed file"

    except OSError:
        # Set remove message
        fileRemoveMessage = "No fine found"

    # Output remove message
    print(csv_path + " - " + fileRemoveMessage)

    try:
        # Create csv writer file object using csv path
        with open(csv_path, 'w', newline='') as csv_writer_file:
            # Create csv writer object
            csv_writer = writer(csv_writer_file)

            # Add header row to csv file
            csv_writer.writerow([constants.IMAGE_PATH, constants.THREAD_COUNT, constants.WIDTH, constants.HEIGHT])

            # Scanning files from database path, return stats of files
            files_valid, files_skipped, files_total = scan_files(csv_writer, image_storage, database_path)

            # Close csv writer object
            csv_writer_file.close()

        # Set processing file outcome
        processing_outcome = f"{database_path}: Files valid - {files_valid}\tFiles skipped - {files_skipped}\tFiles total - {files_total}"

        # Set exection valid flag to true
        execution_valid_flag = True

    except:
        # Set processing file outcome
        processing_outcome = f"{database_path} - invalid, falled scraping process"

        # Set exection valid flag to False
        execution_valid_flag = False

    # Output processing message
    print(processing_outcome)

    # Return exectuion valid flag
    return execution_valid_flag