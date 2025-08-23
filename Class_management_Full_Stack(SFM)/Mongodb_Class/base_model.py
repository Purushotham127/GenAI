from .mongodb import client, database_name

db = client[database_name]

#Student class
class Student():
    name : str = ""
    marks : float = 0.0
    age : int = 0

    # Initializing values
    def __init__(self, name, marks, age):
        self.name = name
        self.marks = marks
        self.age = age

# Base model for MongoDB student documents
class BaseModel:
    collection_name = None

    # Constructor to initialize collection name
    def __init__(self, collection_name):
        self.collection_name = collection_name

    def is_collection_exists(self):
        return self.collection_name in db.list_collection_names()
    
    def add_student(self,student_name, marks, age):
        if not self.is_collection_exists():
            raise ValueError("Collection does not exists.")
        student = Student(student_name=student_name, marks=marks, age=age)
        db[self.collection_name].insert_one(student.__dict__)

    # Get all student names
    def get_all_student_names(self):
        if not self.is_collection_exists():
            raise ValueError("Collection does not exists.")
        student_names = []
        for student in db[self.collection_name].find({},{"_id":0,"name":1}):
            student_names.append(student["name"])
        return student_names

    # Check if student exists
    def is_student_exists(self,student_name):
        try:
            return student_name in self.get_all_student_names()
        except Exception as e:
            raise ValueError(f"Error checking student existence. {e}")

    # Get student details by name
    def get_student_details(self,student_name):
        if not self.is_collection_exists():
            raise ValueError("Collection does not exists.")
        if not self.is_student_exists(student_name):
            raise ValueError(f"Student '{student_name}' does not exist.")
        return db[self.collection_name].find_one({"name":student_name},{"_id":0})

    # Get all student details
    def get_all_student_details(self):
        if not self.is_collection_exists():
            raise ValueError("Collection does not exists.")
        return list(db[self.collection_name].find({},{"_id":0}))

    # Update student marks
    def update_students_marks(self,student_name,marks):
        if not self.is_collection_exists():
            raise ValueError("Collection does not exists.")
        if not self.is_student_exists(student_name):
            raise ValueError(f"Student '{student_name}' does not exist.")
        db[self.collection_name].update_one({"name":student_name},{"$set":{"marks":marks}})

    # Delete student record
    def delete_student(self,student_name):
        if not self.is_collection_exists():
            raise ValueError("Collection does not exists.")
        if not self.is_student_exists(student_name):
            raise ValueError(f"Student '{student_name}' does not exist.")
        db[self.collection_name].delete_one({"name":student_name})