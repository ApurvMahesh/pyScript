from flask import Flask, request, jsonify
import json
import sqlite3


app = Flask(__name__)


def db_connection():
   conn = None
   try:
       conn = sqlite3.connect("events.sqlite")
   except sqlite3.error as e:
       print(e)
   return conn




events_list = [
   {
       "id":0,
       "event_type": "pull_request",
       "event_name": "change_event"
   },


   {
       "id":1,
       "event_type":"release",
       "event_name":"deployment_event"
   },
   {
       "id":2,
       "event_type":"push",
       "event_name":"workflow_event"
   },
   {
       "id":3,
       "event_type": "pull_request_merged",
       "event_name":"deployment_event"
   }
]

@app.route('/events', methods=['GET', 'POST'])
def events():
   conn = db_connection()
   cursor = conn.cursor()


   if request.method == 'GET':
       if len(events_list) > 0:
           # encode list of events in json
           cursor.execute("SELECT * FROM events")
           rows = cursor.fetchall()
           return jsonify(rows)
       else:
           'Event not found', 404
  
   if request.method == 'POST':
       new_event_type = request.form['event_type']
       new_event_name = request.form['event_name']


       sql = """INSERT INTO events (event_type, event_name)
                VALUES (?, ?)"""
       cursor = cursor.execute(sql, (new_event_type, new_event_name))
       conn.commit()
       return f"event with the id: 0 created successfully", 201


       new_obj = {
           'id':iD,
           'event_type': new_event_type,
           'event_name': new_event_name
       }


       events_list.append(new_obj)
       return jsonify(events_list), 201


@app.route('/event/<int:id>', methods=['GET', 'PUT', 'DELETE'])
def single_event_workflow(id):
   if request.method == 'GET':
       for event in events_list:
           if event['id'] -- id:
               return jsonify(event)
           pass
   if request.method == 'PUT':
       sql = """UPDATE event
               SET event_type=?,
                   event_name=?,
               WHERE id=? """
       for event in events_list:
           if event['id'] == id:
               event['event_type'] = request.event['event_type']
               event['event_name'] = request.event['event_name']
               updated_event = {
                   'id':id,
                   'event_type': event['event_type'],
                   'event_name': event['event_name']
               }
               conn.execute(sql, (event_type, event_name, id))
               conn.commit()
               return jsonify(updated_event)




if __name__ == '__main__':
   app.run(host='0.0.0.0', port=int("5000"), debug=True)