# This is a _very simple_ example of a web service that recognizes faces in uploaded images.
# The result is returned as json. For example:
#
# $ curl -XPOST -F "file=@obama2.jpg" http://127.0.0.1:5000

import face_recognition
from flask import Flask, jsonify, request, redirect, render_template, url_for
import numpy as np
import json  #JSON5 MODULE
import os
import datetime
from werkzeug.utils import secure_filename
import pathlib
from glob import glob
import threading

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def upload_form():
    return render_template('upload.html')  #required html file in templates directory
    #return '''
    # <!DOCTYPE html>
    # <html lang="en">
    # <head>
    #     <meta charset="UTF-8">
    #     <title>Title</title>
    #
    #     <script>
    #         window.onload = function(){
    #
    #             //Check File API support
    #             if(window.File && window.FileList && window.FileReader)
    #             {
    #                 var filesInput = document.getElementById("files");
    #
    #                 filesInput.addEventListener("change", function(event){
    #
    #                     var files = event.target.files; //FileList object
    #                     var output = document.getElementById("result");
    #
    #                     for(var i = 0; i< files.length; i++)
    #                     {
    #                         var file = files[i];
    #
    #                         //Only pics
    #                         if(!file.type.match('image'))
    #                           continue;
    #
    #                         var picReader = new FileReader();
    #
    #                         picReader.addEventListener("load",function(event){
    #                             //Read the image
    #
    #                             document.getElementById("result").style.display='block';
    #                             document.getElementById("clear").style.display='block';
    #                             document.getElementById("submit").style.display='block';
    #                             var picFile = event.target;
    #
    #                             var div = document.createElement("div");
    #                             div.id="divframe";
    #                             div.innerHTML = "<img class='thumbnail' src='" + picFile.result + "'" +
    #                                     "title='" + picFile.name + "'/>";
    #
    #                             output.insertBefore(div,null);
    #
    #                         });
    #
    #                          //Read the image
    #                         picReader.readAsDataURL(file);
    #                     }
    #
    #                 });
    #             }
    #             else
    #             {
    #                 console.log("Your browser does not support File API");
    #             }
    #
    #             document.getElementById('files').onclick = function(){
    #                 var e = document.getElementById('divframe');
    #                 e.parentNode.removeChild(e);
    #                 document.getElementById('result').style.display='none';
    #                 document.getElementById('files').value=null;
    #                 document.getElementById('clear').style.display='none';
    #                 document.getElementById("submit").style.display='none';
    #             }
    #
    #             document.getElementById('clear').onclick = function(){
    #                 //var e = document.getElementsByClassName('.thumbnail').parentElement.id;
    #                 var e = document.getElementById('divframe');
    #                 e.parentNode.removeChild(e);
    #                 document.getElementById('result').style.display='none';
    #                 document.getElementById('files').value=null;
    #                 document.getElementById('clear').style.display='none';
    #                 document.getElementById("submit").style.display='none';
    #             }
    #         }
    #
    #     </script>
    #
    #     <style>
    #         body{
    #             font-family: 'Segoe UI';
    #             font-size: 12pt;
    #         }
    #
    #         header h1{
    #             font-size:12pt;
    #             color: #fff;
    #             background-color: #1BA1E2;
    #             padding: 20px;
    #
    #         }
    #         article
    #         {
    #             width: 80%;
    #             margin:auto;
    #             margin-top:10px;
    #         }
    #
    #         .thumbnail{
    #             height: 100px;
    #             margin: 10px;
    #             float: left;
    #         }
    #         #clear{
    #            display:none;
    #         }
    #         #submit{
    #            display:none;
    #         }
    #         #result {
    #             border: 4px dotted #cccccc;
    #             display: none;
    #             float: right;
    #             margin:0 auto;
    #             width: 511px;
    #         }
    #
    #     </style>
    # </head>
    # <body>
    #     <header>
    #     <h1>File API – Image Upload</h1>
    #     </header>
    #     <form method="POST" enctype="multipart/form-data">
    #         <article>
    #         <label for="files">Select multiple files: </label>
    #         <input id="files" type="file" multiple/>
    #         <output id="result" />
    #     </article>
    #     <table>
    #         <tr><td><input type="submit" id="submit" name="submit" value="Upload"></td>
    #             <td><button type="button" id="clear">Clear</button></td>
    #         </tr>
    #     </table>
    #     </form>
    # </body>
    #
    # </html>
    #
    # '''

def write_neuralJson(path, neuralJson):
    #Writing JSON to a file
    # person_dict = {"name": "Bob",
    # "languages": ["English", "Fench"],
    # "married": True,
    # "age": 32
    # }

     with open(path + '/neural.json', 'w') as json_file:
        json.dump(neuralJson, json_file)
    #print(neuralJson)

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file and allowed_file(file):
                r.append(os.path.join(root, file))
    return r

def get_neuraljson(myList):
    message = ""
    path = 'ImagesAttendance'
    neural_json = []

    print("mylist : ", myList)
    for cl in myList:
        neural = []
        i = 1
        label = os.path.splitext(cl)[0]
        myListF = list_files(f'{path}/{cl}')
        for clf in myListF:
            img = face_recognition.load_image_file(clf)
            neural.append(np.array(face_recognition.face_encodings(img)[0]))
            # if len(myListF) == 1:
            #     neural.append(np.array(face_recognition.face_encodings(img)[0]))
            # else:
            #     neural.append(np.array(face_recognition.face_encodings(img)[0]))

        descriptors = np.array(neural).tolist()
        if descriptors == "":
            if pathlib.Path(f'{path}/{cl}').exists():
                pathlib.Path(f'{path}/{cl}').rmdir(parents=True, exist_ok=True)
            message = "Registration Failed"
        else:
            message = "Face Registration Successfully"
        neural_json.append({"label": label, "descriptors": descriptors})

    #print(neural_json)
    write_neuralJson(f'{path}/{cl}', neural_json)
    return message

def merge_neuraljson():
    data = []

    for f in glob("./ImagesAttendance" + "/**/neural.json", recursive=True):

        with open(f, 'r') as infile:
            #data.append(json.load(infile))
            data.extend(json.load(infile))


    with open("neural.json", 'w') as outfile:
        json.dump(data, outfile)
    print("neural update done...")


@app.route('/', methods=['POST'])
def upload_image():
    # if 'files[]' not in request.files:
    #     flash('No file part')
    #     return redirect(request.url)
    #
    empid = request.form['empid'].strip()
    empname = request.form['empname'].strip()

    path = 'ImagesAttendance/' + empname + '@' + empid + '/'
    dirname = empname + '@' + empid

    if pathlib.Path(f'{path}').exists():
        dt = datetime.datetime.now()
        timestamp = dt.strftime("%Y") + "/" + dt.strftime("%m") + "/" + dt.strftime("%d") + " " + dt.strftime(
            "%H") + ":" + dt.strftime("%M") + ":" + dt.strftime("%S")
        # Return the result as json
        result = {"result": {
            "empid": empid,
            "name": empname,
            "message": "Record already exist.",
            "timestamp": timestamp
        }}
        return jsonify(result)


    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    files = request.files.getlist('files[]')
    file_names = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(path, filename))
        #else:
		#	flash('Allowed image types are -> png, jpg, jpeg, gif')
		#	return redirect(request.url)
    print(path)
    print(dirname)
    print("file list : ", request.files.getlist('files[]'))
    if not request.files.getlist('files[]'):
        if pathlib.Path(path).exists():
            pathlib.Path(path).rmdir()
        result_return = "Registration Failed"
    else:
        result_return = get_neuraljson([dirname])

    dt = datetime.datetime.now()
    timestamp = dt.strftime("%Y") + "/" + dt.strftime("%m") + "/" + dt.strftime("%d") + " " + dt.strftime(
        "%H") + ":" + dt.strftime("%M") + ":" + dt.strftime("%S")

    # return render_template('upload.html', dirname=dirname, filenames=file_names)

    # Return the result as json
    result = {"result": {
        "empid": empid,
        "name": empname,
        "message": result_return,
        "timestamp": timestamp
    }}
    return jsonify(result)



@app.route('/display/<filename>')
def display_image(dirname, filename):
    #app.config['UPLOAD_FOLDER'] = os.path.join(os.environ['HOME'], 'ImagesAttendance', '3434@raja')
    #print('upload_image filename: ' , filename) #+ path.rstrip(path[:-1])
    #return redirect(url_for('static', filename='/ImagesAttendance/3434@raja/20190417_093356.jpg'), code=301)
    #return render_template("upload.html", dirname=dirname, filename=filename)
    # Return the result as json
    neuralinfoArray = dirname.split("@")
    name = neuralinfoArray[0]
    empid = neuralinfoArray[1]

    dt = datetime.datetime.now()
    timestamp = dt.strftime("%Y") + "/" + dt.strftime("%m") + "/" + dt.strftime("%d") + " " + dt.strftime(
        "%H") + ":" + dt.strftime("%M") + ":" + dt.strftime("%S")
    result = {"result": {
        "empid": empid,
        "name": empname,
        "message": message,
        "timestamp": timestamp
    }}
    return jsonify(result)


def job():
    dt = datetime.datetime.now()
    thread_timestamp = dt.strftime("%Y") + "/" + dt.strftime("%m") + "/" + dt.strftime("%d") + " " + dt.strftime(
        "%H") + ":" + dt.strftime("%M") + ":" + dt.strftime("%S")

    merge_neuraljson()
    print("I'm working..." + thread_timestamp)
    threading.Timer(60.0, job).start()  # called every minute


#merge_neuraljson()


# if __name__ == "__main__":
#     app.run()

if __name__ == "__main__":
    job()
    app.run(host='0.0.0.0', port=5001, debug=True)

