using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

#region Tensorflow Libraries
using Tensorflow;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
#endregion

namespace ObjectDetectionWithTensorflow
{
    public class ObjectDetection : IDisposable
    {
        string modelDir = "ssd_mobilenet_v1_coco_2018_01_28";
        string imageDir = "images";
        string pbFile = "frozen_inference_graph.pb";
        float MIN_SCORE = 0.5f;
        Graph graph;
        Session session;

        public ObjectDetection()
        {
            tf.compat.v1.disable_eager_execution();

            #region Get Tensorflow Model From Web If No Exists
            // get model file
            string url = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz";
            Web.Download(url, modelDir, "ssd_mobilenet_v1_coco.tar.gz");

            Compress.ExtractTGZ(Path.Join(modelDir, "ssd_mobilenet_v1_coco.tar.gz"), "./");

            // download the pbtxt file
            url = $"https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt";
            Web.Download(url, modelDir, "mscoco_label_map.pbtxt");
            #endregion

            #region Import model to graph
            graph = new Graph().as_default();
            graph.Import(Path.Join(modelDir, pbFile));

            #endregion

            #region Run Session
            session = tf.Session(graph);
            #endregion
        }

        public void Dispose()
        {
            graph.Dispose();
            session.Dispose();
        }

        public Bitmap Predict(Bitmap originalImage)
        {

                byte[] BitmapToByteArray(Bitmap bitmap, PixelFormat pixelFormat = PixelFormat.DontCare)
                {
                    BitmapData bitmapData = null;

                    try
                    {
                        bitmapData = bitmap.LockBits(
                            new Rectangle(0, 0, bitmap.Width, bitmap.Height),
                            ImageLockMode.ReadOnly,
                            pixelFormat == PixelFormat.DontCare ? bitmap.PixelFormat : pixelFormat
                        );

                        var size = bitmapData.Stride * bitmap.Height;
                        var bytes = new byte[size];
                        var ptr = bitmapData.Scan0;
                        Marshal.Copy(ptr, bytes, 0, size);

                        return bytes;
                    }
                    finally
                    {
                        if (bitmapData != null)
                            bitmap.UnlockBits(bitmapData);
                    }
                }

                #region Prediction
                var pixelFormat = PixelFormat.Format24bppRgb;  // my model assumes RGB images
                var bytes = BitmapToByteArray(originalImage, pixelFormat);
                Tensor image = new NDArray(bytes, new Shape(1, originalImage.Height, originalImage.Width, 3), TF_DataType.TF_UINT8);
                //Tensor image = new NDArray(originalImage.Bytes, new Shape(1, originalImage.Height, originalImage.Width, 3), TF_DataType.TF_UINT8);
                Tensor tensorNum = graph.OperationByName("num_detections");
                Tensor tensorBoxes = graph.OperationByName("detection_boxes");
                Tensor tensorScores = graph.OperationByName("detection_scores");
                Tensor tensorClasses = graph.OperationByName("detection_classes");
                Tensor imgTensor = graph.OperationByName("image_tensor");
                Tensor[] outTensorArr = new Tensor[] { tensorNum, tensorBoxes, tensorScores, tensorClasses };
                var detectionResults = session.run(outTensorArr, new FeedItem(imgTensor, image));
                #endregion

                #region Build Output Image
                PbtxtItems pbTxtItems = PbtxtParser.ParsePbtxtFile(Path.Join(modelDir, "mscoco_label_map.pbtxt"));
                var scores = detectionResults[2].ToArray<float>();
                var boxes = detectionResults[1].ToArray<float>();
                var id = np.squeeze(detectionResults[3]).ToArray<float>();
                for (int i = 0; i < scores.Length; i++)
                {
                    float score = scores[i];
                    if (score > MIN_SCORE)
                    {
                        float top = boxes[i * 4] * originalImage.Height;
                        float left = boxes[i * 4 + 1] * originalImage.Width;
                        float bottom = boxes[i * 4 + 2] * originalImage.Height;
                        float right = boxes[i * 4 + 3] * originalImage.Width;

                        string name = pbTxtItems.items.Where(w => w.id == id[i]).Select(s => s.display_name).FirstOrDefault();

                        using (Graphics grap = Graphics.FromImage(originalImage))
                        {
                            grap.DrawRectangle(new Pen(Color.White)
                            {
                                Color = Color.White,
                                Width = 5
                            }, left, top, (right - left), (bottom - top));
                            grap.DrawString(name, new Font(FontFamily.GenericSansSerif, 12), new SolidBrush(Color.Green), left + 5, top + 5);
                        }
                    }
                }
                #endregion

                return originalImage;
            
        }
    }
}
