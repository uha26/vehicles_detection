from flask import Flask, request, jsonify
from flask_restplus import Api, Resource, fields
from werkzeug.utils import secure_filename
import torch
import os

# Initialize Flask and Swagger
app = Flask(__name__)
api = Api(app, version='1.0', title='Vehicle Detection API',
          description='API for vehicle detection using YOLOv5')

# Define upload and output directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

# Define Swagger model for image upload
upload_parser = api.parser()
upload_parser.add_argument('file', type='file', required=True, help='Image file for detection')

# Helper function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create an API resource for uploading and detecting vehicles
@api.route('/upload')
class VehicleDetection(Resource):
    def post(self):
        # Parse the request for the file
        args = upload_parser.parse_args()
        file = args['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Perform inference with YOLOv5
            results = model(file_path)
            output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'result_' + filename)

            # Save the result image
            results.save(output_image_path)

            # Return the result image URL in response
            return jsonify({
                'message': 'Vehicle detection complete',
                'result_image': f"/{OUTPUT_FOLDER}/result_{filename}"
            })
        else:
            return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    # Ensure upload and output directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Run the Flask app
    app.run(debug=True)
