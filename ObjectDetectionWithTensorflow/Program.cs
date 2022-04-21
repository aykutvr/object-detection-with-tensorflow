using ObjectDetectionWithTensorflow;
using OpenCvSharp;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

Console.WriteLine("Hello, World!");



using (OpenCvSharp.VideoCapture capture = new OpenCvSharp.VideoCapture(0))
{
    if (!Directory.Exists(Path.Combine(Environment.CurrentDirectory, "images")))
        Directory.CreateDirectory(Path.Combine(Environment.CurrentDirectory, "images"));

    int sleepTime = (int)Math.Round(1000 / capture.Fps);
    var imageMat = new Mat();

    using (ObjectDetection detection = new ObjectDetection())
    {
        while (true)
        {
            DateTime predictionStart = DateTime.Now;

            #region Read Image From Webcam
            capture.Read(imageMat);
            if (imageMat.Empty())
                continue;
            #endregion


            using (var ms = imageMat.ToMemoryStream(".jpeg"))
            {
                var predictedImage = detection.Predict(new Bitmap(ms));
                predictedImage.Save(Path.Combine(Environment.CurrentDirectory, "images", $"{DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss-ffffff")}_PREDICTED.jpg"), ImageFormat.Jpeg);
            }

            DateTime predictionEnd = DateTime.Now;

            Console.WriteLine($"FPS : { 1 / (predictionEnd - predictionStart).TotalSeconds}");
        }
    }

}
