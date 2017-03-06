# code: UTF-8
import tornado.ioloop
import tornado.web
import uuid
import Image
import StringIO
import os
import commands
import guess
import json


class PredictHandler(tornado.web.RequestHandler):
     def get(self):
	    self.render("index.html")

     def post(self):  
        if self.request.files:  
            file_name = "%s" % uuid.uuid1()
            print 'file_name',file_name
            file_raw = self.request.files["file"][0]["body"]
            usr_home = os.path.expanduser('~')
            file_name = usr_home+"/tensorflow/static/tmp/m_%s.jpg" % file_name
            fin = open(file_name,"w")
            print "success to open file"  
            fin.write(file_raw)  
            fin.close()
            print "use tensorflow"
            age = guess.guessAge(file_name)
            print 'guess age is ', age
            feeds_json = json.dumps(age)
            self.set_header('Content-Type', 'application/json; charset=UTF-8')
            self.write(feeds_json)
            self.finish()
            #output age  score and gender
            #gender = guess.guessGender(image_file)



application = tornado.web.Application([
    (r"/predict", PredictHandler),
])

if __name__ == "__main__":
    application.listen(8882)
    tornado.ioloop.IOLoop.instance().start()
