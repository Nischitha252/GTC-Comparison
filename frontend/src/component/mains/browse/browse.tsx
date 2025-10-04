// browse.tsx
import React, { useEffect, useRef, useState } from "react";
import "./browse.css"; // Create a corresponding CSS file for styling
import UploadIcon from "assets/icon/ip-upload-icon";

interface BrowseFileProps {
  inputId: string;
  multiple?: boolean;
  disabled: boolean;
  reset?: boolean;
  activateFileName?: boolean;
  onFileUpload: (files: FileList | null) => void;
  accept: string;
  format: boolean;
  api_message: (value: string[]) => void;
}

const BrowseFile: React.FC<BrowseFileProps> = ({
  inputId,
  multiple,
  disabled,
  reset,
  activateFileName,
  onFileUpload,
  accept,
  format,
  api_message,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isSelectedFileName, setIsSelectedFileName] = useState<string[]>([]);

  // const handleFileUpload = (files: FileList | null) => {
  //   if (files && files.length > 0) {
  //     const fileLists = Array.from(files);
  //     const fileNames = fileLists.map((file) => file.name);
  //     setIsSelectedFileName(fileNames);
  //     // setIsUploadClicked(false);
  //     onFileUpload(files);
  //   } else {
  //     onFileUpload(null);
  //   }
  // };

  const handleFileUpload = (files: FileList | null) => {
    if (files && files.length > 0) {
      const acceptedTypes = format
        ? [".pdf", ".docx"]
        : [".pdf", ".docx", ".doc", ".xls", ".xlsx", ".csv", ".pptx"];
      // Check for valid files
      const invalidFiles = Array.from(files).filter((file) => {
        const fileExtension = `.${file.name.split(".").pop()?.toLowerCase()}`;
        return !acceptedTypes.includes(fileExtension);
      });

      // If there are invalid files, throw an error message
      if (invalidFiles.length > 0) {
        // Optionally, you can set an error message in state here
        // setApiMessage('Invalid file type(s): ' + invalidFiles.map(file => file.name).join(', '));
        api_message(["Upload File is not a valid", "orange"]);
        setTimeout(() => {
          api_message([]);
        }, 3000);
        return; // Stop further processing
      }

      // If all files are valid, proceed with setting the file names and uploading
      const validFileNames = Array.from(files).map((file) => file.name);
      setIsSelectedFileName(validFileNames);
      onFileUpload(files);
    } else {
      onFileUpload(null);
    }
  };

  const handleFileInputChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const files = event.target.files;

    // Ensure only one file is selected
    if (files && files.length > 1) {
      api_message(["Only one file can be selected", "orange"]);
      setTimeout(() => {
        api_message([]);
      }, 3000);
      event.target.value = ""; // Clear the input
      return;
    }

    handleFileUpload(files); // Proceed with handling the single file
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    handleFileUpload(files);
  };

  const getFileName = (fileName: string) => {
    const [name, extension] = fileName.split(".");
    return name.length > 16 ? `${name.slice(0, 16)}...${extension}` : fileName;
  };

  useEffect(() => {
    if (reset) {
      setIsSelectedFileName([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      onFileUpload(null);
    }
  }, [reset]);

  return (
    <div>
      <div
        className="browseBox"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <label
          htmlFor={inputId}
          className={`browseFile ${disabled ? "disabled" : ""}`}
        >
          <UploadIcon
            className={`abbIcon  ${disabled ? "disabled" : ""}`}
            size="large"
          />
          <span>Browse files</span>
        </label>

        <input
          type="file"
          id={inputId}
          accept={accept}
          multiple
          disabled={disabled}
          style={{ display: "none" }}
          onChange={handleFileInputChange}
        />
      </div>

      {activateFileName && isSelectedFileName.length > 0 && (
        <div className="selectedFileName">
          {/* <h3>Selected files:</h3> */}
          {isSelectedFileName.map((fileName, index) => (
            <span key={index}>{getFileName(fileName)}</span>
          ))}
        </div>
      )}
    </div>
  );
};

export default BrowseFile;
