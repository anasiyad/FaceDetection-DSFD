import cv2
from os import path 
khan = cv2.imread('img_2.png')
# الصورة الثانية
kids = cv2.imread('img_2.png')

# عرفنا متغير لاستخدام ملف json ,, وظيفته ملف جاهز للتعرف على الأوجه
# haarcascade_frontalface_default.xml اسم الملف الخاص بالتعرف على الأوجه
#    ملاحظة للتعامل مع ملفات خارجية ك ملفات  لا نستطيع استقبالها مباشرة لا من خلال مكتبة os وهنا أتي دورها في التعامل مع ملفات josn
xml_classifier = path.join(path.dirname(cv2.__file__),
                           "data", "haarcascade_frontalface_default.xml")

# هنا أنشائنا ميثود وظيفتها التعرف على الأشخاص وخاصة الأوجه
def detect_faces(image):
    # هذا السطر وظيفته تحويل الصورة من BGR إلى RGB
    # RGB السبب لان model لا يستطيع الكشف عن الأشخاص إلا من خلال التعرف القناة اللونية
    # ونحن نرى ب BGR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #  هنا استخدمنا المتغير الخاص بملف ال josn الخاص بالتعرف على الاوجه "xml_classifier" وعملنا استدعاء في الميثود بعد تحويل القناة اللونية
    face_calssifier = cv2.CascadeClassifier(xml_classifier)
    # هنا دمجنا جميع الوظائف بمتغير واحد "rects" ويحتوى المتغير على face_calssifier و التدرج الرمادي Gray وscaleFactor و minNeighbors و minSize في ومتغير واحد
    rects = face_calssifier.detectMultiScale(image=gray,
                                             scaleFactor=1.15,
                                             minNeighbors=5,
                                             minSize=(30, 30))
# هنا بعد سيتم ارجاع القيمة بمتغير rects للوظائف التي تنفذها الميثود detect_faces
    return rects

# في هذه الميثود سوف نرسم الأطار الخاص بتحديد الأوجه في الصورة
# وتستقبل الميثود  متغير image الي عرفناه فوق و متغير reacts الي حينفذ عملية التعرف علي الوجه
# عرفنا متغير rects عشان حربطو مع ميثود draw وال title بقيمة none
def draw(image, rects, title=None):
    # هنا أعطينا حجم الإطار في الصورة وهو 30 في 30 الذي سيتم رسمه على الوجه المتعرف عليه
    print("=" * 30)
    # هنا استخدمنا دالة len بحيث يعطيني عدد الاشخاص بالصورة في output
    print("i found {} people.".format(len(rects)).title())
    print("=" * 30)
# من خلال for ال model حيمشي على جميع الاوجه المتعرف عليها ووضع أطار علي جميعهم
    for x, y, w, h in rects:
        #في هذا السطر سوف اعرف استدعي image الذي يحمل الصورة وهنا في المتغير سوف يضع اطار علي جميع الاوجه في الصورة وبدون ال for loop مش حيحدد غير شخص واحد
        # وتتيح خاصية rectangle اجراء عمليه جمع لتحديد مجموعة الاوجه,, (0, 255, 0) وهيا الخاص بلون الإطار وهو الاخضر ,,  2 والتي لعرض الخط اي border خط الإطار
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
# شرط اذا تواجد عنوان بالصورة اضفه والشرط هنا True
    if title:
        cv2.imshow(title, image)
        cv2.waitKey(0)


cv2.imshow("Khan", khan)
cv2.waitKey(0)
# هنا استدعيت ميثود  draw  الخاصة برسم الإطار حول الوجه وبداخلها ميثود detect_faces واعطيتها متغير الصورة التي قرائها  وعنوان للصورة
draw(khan, detect_faces(khan), "Khan")

cv2.imshow("Kids", kids)
cv2.waitKey(0)
draw(kids, detect_faces(kids), "Kids")
