﻿using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ObjectDetectionWithTensorflow
{
    public class PbtxtItem
    {
        public string name { get; set; }
        public int id { get; set; }
        public string display_name { get; set; }
    }
    public class PbtxtItems
    {
        public List<PbtxtItem> items { get; set; }
    }

    public class PbtxtParser
    {
        public static PbtxtItems ParsePbtxtFile(string filePath)
        {
            string line;
            string newText = "{\"items\":[";

            using (System.IO.StreamReader reader = new System.IO.StreamReader(filePath))
            {

                while ((line = reader.ReadLine()) != null)
                {
                    string newline = string.Empty;

                    if (line.Contains("{"))
                    {
                        newline = line.Replace("item", "").Trim();
                        //newText += line.Insert(line.IndexOf("=") + 1, "\"") + "\",";
                        newText += newline;
                    }
                    else if (line.Contains("}"))
                    {
                        newText = newText.Remove(newText.Length - 1);
                        newText += line;
                        newText += ",";
                    }
                    else
                    {
                        newline = line.Replace(":", "\":").Trim();
                        newline = "\"" + newline;// newline.Insert(0, "\"");
                        newline += ",";

                        newText += newline;
                    }

                }

                newText = newText.Remove(newText.Length - 1);
                newText += "]}";

                reader.Close();
            }

            PbtxtItems items = JsonConvert.DeserializeObject<PbtxtItems>(newText);

            return items;
        }
    }
}
