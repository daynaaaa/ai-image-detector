import React from 'react';
import { useState } from "react";
import { useDropzone } from "react-dropzone";

export default function ImageUpload() {
    const [image, setImage] = useState<string | null>(null);

    const { getRootProps, getInputProps, open } = useDropzone({
        accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp"] },
        onDrop: (acceptedFiles) => {
        setImage(URL.createObjectURL(acceptedFiles[0]));
        },
        noClick: true,
    });
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
            <p className="text-gray-600">Drag & drop or click to upload</p>
          )}
        </div>
        <button 
            onClick={open}
            className="px-4 py-2 bg-blue-500 text-white rounded-md shadow">
          Upload Image
        </button>

        {image && (
        <button
          onClick={() => setImage(null)} 
          className="mt-2 px-4 py-2 bg-red-500 text-white rounded-md shadow"
        >
          Remove Image
        </button>
        )}
      </div>
    )
}
