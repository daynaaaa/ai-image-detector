import React from 'react';
import { useState } from "react";
import { useDropzone } from "react-dropzone";

export default function ImageUpload() {
    const [image, setImage] = useState<string | null>(null);
    const [prediction, setPrediction] = useState<string | null>(null);
    const [confidence, setConfidence] = useState<number | null>(null);
    const [loading, setLoading] = useState(false);


    const { getRootProps, getInputProps, open } = useDropzone({
        accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp"] },
        onDrop: async (acceptedFiles) => {
          const file = acceptedFiles[0];
          setImage(URL.createObjectURL(acceptedFiles[0]));
          await handlePrediction(file);
        },
        noClick: true,
    });

    const handlePrediction = async (file: File) => {
      setLoading(true);
      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();
        setPrediction(data.prediction);
        setConfidence(data.confidence);
      } catch (error) {
        console.error('The error is:', error);
      } finally {
        setLoading(false);
      }
    };

    return (
        <div className="flex flex-col items-center gap-4 p-6 bg-gray-100 rounded-xl shadow-md">
        <div
          {...getRootProps()}
          className="w-64 h-40 flex items-center justify-center border-2 border-dashed border-gray-400 rounded-lg cursor-pointer bg-white"
        >
          <input {...getInputProps()} />
          {image ? (
            <img src={image} alt="Preview" className="max-h-full rounded-md" />
          ) : (
            <p className="text-gray-600">Drag & drop or click to upload an image to determine if it is AI-Generated or not</p>
          )}
        </div>
        <button 
            onClick={open}
            className="px-4 py-2 bg-blue-500 text-white rounded-md shadow">
          Upload Image
        </button>

        {loading && <p>Analyzing image ...</p>}

        {prediction && (
          <div className='mt-4 text-center'>
            <p className="font-bold">Prediction: {prediction}</p>
            <p>Confidence: {(confidence! * 100).toFixed(2)}%</p>
          </div>
        )}

        {image && (
          <button
            onClick={() => {
              setImage(null);
              setPrediction(null);
              setConfidence(null);
            }} 
          className="mt-2 px-4 py-2 bg-red-500 text-white rounded-md shadow"
        >
          Remove Image
        </button>
        )}
      </div>
    )
}
