import React, { useEffect, useRef, useState } from "react";
import "./commercialMultipleBrowse.css";
import UploadIcon from "assets/icon/ip-upload-icon";
import { HandleFileUploadMultipleExcelExtraction } from "utils/helperFunction";
import { MultiSelect } from "primereact/multiselect";

interface BrowseFileProps {
  inputId: string;
  disabled: boolean;
  reset?: boolean;
  activateFileName?: boolean;
  accept: string;
  sendingStoreMultipleFileToParent: React.Dispatch<
    React.SetStateAction<File[]>
  >;
  sendingSelectedMultipleColumnValues: (values: {
    [key: string]: string[];
  }) => void;
  excludedColumn: string[];
  api_message: (message: string[]) => void;
}

const CommercialBrowseMultipleFile: React.FC<BrowseFileProps> = ({
  inputId,
  disabled,
  reset,
  activateFileName,
  accept,
  sendingStoreMultipleFileToParent,
  sendingSelectedMultipleColumnValues,
  excludedColumn,
  api_message,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [storeFiles, setStoreFiles] = useState<File[]>([]);
  const {
    handleFileUploadExcelIndividual,
    storeExcelColumn,
    removeObjectProperties,
    resetColumn,
  } = HandleFileUploadMultipleExcelExtraction();

  const [selectedColumnValues, setSelectedColumnValues] = useState<{
    [key: string]: string[];
  }>({});

  const storingFile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const acceptedTypes = [".xlsx"];

      const invalidFiles = Array.from(files).filter((file) => {
        const fileExtension = `.${file.name.split(".").pop()?.toLowerCase()}`;
        return !acceptedTypes.includes(fileExtension);
      });

      if (invalidFiles.length > 0) {
        api_message(["Upload File is not a valid format", "orange"]);
        setTimeout(() => {
          api_message([]);
        }, 3000);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
        return;
      }

      handleFileUploadExcelIndividual(event);

      const newFiles = Array.from(files);
      const uniqueFiles = newFiles.filter(
        (file) =>
          !storeFiles.some((existingFile) => existingFile.name === file.name)
      );
      // setStoreFiles((prevFiles) => [...prevFiles, ...uniqueFiles]);
      // Merge files and limit to the latest 5
      const updatedFiles = [...storeFiles, ...uniqueFiles];
      const limitedFiles = updatedFiles.slice(-5); // Retain only the last 5 files

      // Update the state with the limited files
      setStoreFiles(limitedFiles);
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  useEffect(() => {
    sendingStoreMultipleFileToParent(storeFiles);
  }, [storeFiles, sendingStoreMultipleFileToParent]);

  useEffect(() => {
    if (reset) {
      setStoreFiles([]);
      resetColumn();
      setSelectedColumnValues({});
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }, [reset, resetColumn]);

  const removeFile = (name: string) => {
    setStoreFiles((prevFiles) =>
      prevFiles.filter((file) => file.name !== name)
    );
    setSelectedColumnValues((prevValues) => {
      const newValues = { ...prevValues };
      delete newValues[name];
      return newValues;
    });
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleColumnDropDownValueChange = (
    fileName: string,
    e: { value: { name: string }[] }
  ) => {
    const transformedSelection = e.value
      .map((item) => item.name)
      .filter((name) => !excludedColumn.includes(name));
    setSelectedColumnValues((prevState) => ({
      ...prevState,
      [fileName]: transformedSelection,
    }));
  };

  useEffect(() => {
    setSelectedColumnValues((prevState) => {
      const newState = { ...prevState };
      Object.keys(newState).forEach((fileName) => {
        newState[fileName] = newState[fileName].filter(
          (column) => !excludedColumn.includes(column)
        );
      });
      return newState;
    });
  }, [excludedColumn]);

  useEffect(() => {
    sendingSelectedMultipleColumnValues(selectedColumnValues);
  }, [selectedColumnValues, sendingSelectedMultipleColumnValues]);

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files && files.length > 0) {
      const changeEvent = {
        target: {
          files: files,
        },
      } as React.ChangeEvent<HTMLInputElement>;
      storingFile(changeEvent);
    }
  };

  return (
    <div style={{ marginTop: "-15px" }}>
      <div
        className="browseBox"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <label
          htmlFor={inputId}
          className={`browseFile ${
            disabled || storeFiles?.length >= 5 ? "disabled" : ""
          }`}
        >
          <UploadIcon
            className={`abbIcon  ${
              disabled || storeFiles?.length >= 5 ? "disabled" : ""
            }`}
            size="large"
          />
          <span>Browse files</span>
        </label>

        <input
          type="file"
          id={inputId}
          accept={accept}
          ref={fileInputRef}
          multiple
          disabled={disabled || storeFiles?.length >= 5}
          style={{ display: "none" }}
          onChange={storingFile}
        />
      </div>

      {activateFileName && storeFiles?.length > 0 && (
        <div className="mfSelectedFileName">
          {storeFiles.map((file, index) => (
            <div key={index}>
              <div className="cont">
                <p className="mfSelectedFileNameSpan">
                  {file.name.length > 16
                    ? `${file.name.slice(0, 16)}...${file.name
                        .split(".")
                        .pop()}`
                    : file.name}
                </p>
                <p
                  className="sendIcon"
                  style={{ cursor: "pointer" }}
                  onClick={() => {
                    removeFile(file.name);
                    removeObjectProperties(file.name);
                  }}
                >
                  &times;
                </p>
              </div>
              <div className="selectLayoutEx-cols">
                <label
                  htmlFor="selectLayoutExcelColumn"
                  style={{ marginLeft: "10px" }}
                  className="comMBLabelHeading"
                >
                  Select layout Excel column
                </label>
                <MultiSelect
                  value={
                    selectedColumnValues[file.name]?.map((value) => ({
                      name: value,
                    })) || []
                  }
                  onChange={(e) =>
                    handleColumnDropDownValueChange(file.name, e)
                  }
                  options={(storeExcelColumn[file.name] || [])
                    .filter((value: string) => !excludedColumn.includes(value))
                    .map((value: string) => ({ name: value }))}
                  optionLabel="name"
                  placeholder="Select Excel column"
                  maxSelectedLabels={2}
                  style={{ maxWidth: "200px" }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default CommercialBrowseMultipleFile;
