// browse.tsx
import React, { useEffect, useRef, useState } from "react";
import "./commercialMultipleBrowse.css"; // Create a corresponding CSS file for styling
import UploadIcon from "assets/icon/ip-upload-icon";
import { HandleFileUploadMultipleExcelExtraction } from "utils/helperFunction";
import { MultiSelect } from "primereact/multiselect";

interface BrowseFileProps {
  inputId: string;
  disabled?: boolean;
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
  set_api_message: (message: string[]) => void;
  formatValue: string;
}

const FacilitiesBrowseMultipleFile: React.FC<BrowseFileProps> = ({
  inputId,
  disabled,
  reset,
  activateFileName,
  accept,
  sendingStoreMultipleFileToParent,
  sendingSelectedMultipleColumnValues,
  excludedColumn,
  set_api_message,
  formatValue,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isSelectedFileName, setIsSelectedFileName] = useState<string[]>([]);
  const [storeFiles, setStoreFiles] = useState<File[]>([]);
  const {
    handleFileUploadExcelIndividual,
    storeExcelColumn,
    removeObjectProperties,
    resetColumn,
  } = HandleFileUploadMultipleExcelExtraction();

  // future usecase

  // const storingFile = (event: React.ChangeEvent<HTMLInputElement>) => {
  //   const files = event.target.files;
  //   handleFileUploadExcelIndividual(event);
  //   if (files) {
  //     const newFiles = Array.from(files);
  //     const uniqueFiles = newFiles.filter(
  //       (file) =>
  //         !storeFiles.some((existingFile) => existingFile.name === file.name)
  //     );
  //     setStoreFiles((prevFiles) => [...prevFiles, ...uniqueFiles]);
  //   }

  //   if (fileInputRef.current) {
  //     fileInputRef.current.value = "";
  //   }
  // };
  //future usecase
  // const storingFile = (event: React.ChangeEvent<HTMLInputElement>) => {
  //   handleFileUploadExcelIndividual(event); // Changed from handleFileUploadExcelIndividual
  //   const files = event.target.files;
  //   if (files) {
  //     const newFiles = Array.from(files);
  //     const uniqueFiles = newFiles.filter(
  //       (file) =>
  //         !storeFiles.some((existingFile) => existingFile.name === file.name)
  //     );
  //     setStoreFiles((prevFiles) => [...prevFiles, ...uniqueFiles]);
  //   }

  //   if (fileInputRef.current) {
  //     fileInputRef.current.value = "";
  //   }
  // };
  // const storingFile = (event: React.ChangeEvent<HTMLInputElement>) => {
  //   const files = event.target.files;
  //   if (files && files.length > 0) {
  //     const acceptedTypes = [".pdf", ".xlsx"];

  //     const invalidFiles = Array.from(files).filter((file) => {
  //       const fileExtension = `.${file.name.split(".").pop()?.toLowerCase()}`;
  //       return !acceptedTypes.includes(fileExtension);
  //     });

  //     if (invalidFiles.length > 0) {
  //       // Invalid file type(s)
  //       set_api_message(["Upload File is not a valid format", "orange"]);
  //       setTimeout(() => {
  //         set_api_message([]);
  //       }, 3000);
  //       // Reset the file input
  //       if (fileInputRef.current) {
  //         fileInputRef.current.value = "";
  //       }
  //       return;
  //     }

  //     // All files are valid, proceed with processing
  //     handleFileUploadExcelIndividual(event);

  //     const newFiles = Array.from(files);
  //     const uniqueFiles = newFiles.filter(
  //       (file) =>
  //         !storeFiles.some((existingFile) => existingFile.name === file.name)
  //     );
  //     setStoreFiles((prevFiles) => [...prevFiles, ...uniqueFiles]);
  //   }

  //   if (fileInputRef.current) {
  //     fileInputRef.current.value = "";
  //   }
  // };
  // Add this state at component level

  // Add this state at component level
  // const storingFile = (event: React.ChangeEvent<HTMLInputElement>) => {
  //   const files = event.target.files;
  //   if (files && files.length > 0) {
  //     const acceptedTypes = [".pdf", ".xlsx",".docx"];

  //     // Get first file's format of current selection
  //     const currentFileExtension = `.${files[0].name
  //       .split(".")
  //       .pop()
  //       ?.toLowerCase()}`;

  //     // Check if first file is valid format
  //     if (!acceptedTypes.includes(currentFileExtension)) {
  //       set_api_message([
  //         "Please upload only PDF or Excel (.xlsx) files",
  //         "orange",
  //       ]);
  //       setTimeout(() => {
  //         set_api_message([]);
  //       }, 3000);
  //       if (fileInputRef.current) {
  //         fileInputRef.current.value = "";
  //       }
  //       return;
  //     }

  //     // If there are already stored files, check if new files match the format
  //     // if (storeFiles.length > 0) {
  //     //   const existingFileExtension = `.${storeFiles[0].name
  //     //     .split(".")
  //     //     .pop()
  //     //     ?.toLowerCase()}`;

  //     //   if (currentFileExtension !== existingFileExtension) {
  //     //     const formatType =
  //     //       existingFileExtension === ".pdf" ? "PDF" : "Excel (.xlsx)";
  //     //     set_api_message([
  //     //       `You've already uploaded ${formatType} files. Please continue with the same format.`,
  //     //       "orange",
  //     //     ]);
  //     //     setTimeout(() => {
  //     //       set_api_message([]);
  //     //     }, 3000);
  //     //     if (fileInputRef.current) {
  //     //       fileInputRef.current.value = "";
  //     //     }
  //     //     return;
  //     //   }
  //     // }

  //     // Check if ALL files in current selection match the format
  //     // const invalidFiles = Array.from(files).filter((file) => {
  //     //   const fileExtension = `.${file.name.split(".").pop()?.toLowerCase()}`;
  //     //   return fileExtension !== currentFileExtension;
  //     // });

  //     // if (invalidFiles.length > 0) {
  //     //   const formatType =
  //     //     currentFileExtension === ".pdf" ? "PDF" : "Excel (.xlsx)";
  //     //   set_api_message([
  //     //     `All files must be pdf or excel format. Mixed formats are not allowed.`,
  //     //     "orange",
  //     //   ]);
  //     //   setTimeout(() => {
  //     //     set_api_message([]);
  //     //   }, 3000);
  //     //   if (fileInputRef.current) {
  //     //     fileInputRef.current.value = "";
  //     //   }
  //     //   return;
  //     // }

  //     // All files are valid and same format, proceed
  //     // handleFileUploadExcelIndividual(event);

  //     const newFiles = Array.from(files);
  //     const uniqueFiles = newFiles.filter(
  //       (file) =>
  //         !storeFiles.some((existingFile) => existingFile.name === file.name)
  //     );
  //     setStoreFiles((prevFiles) => [...prevFiles, ...uniqueFiles]);
  //   }

  //   if (fileInputRef.current) {
  //     fileInputRef.current.value = "";
  //   }
  // };

  const storingFile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;

    // Define accepted types based on formatValue
    let acceptedTypes: string[] = [];
    if (formatValue === "format1") {
      acceptedTypes = [".pdf", ".xlsx", ".docx"];
    } else if (formatValue === "format2") {
      acceptedTypes = [".xlsx"];
    }

    if (files && files.length > 0) {
      // Get the first file's format of the current selection
      const currentFileExtension = `.${files[0].name
        .split(".")
        .pop()
        ?.toLowerCase()}`;

      // Check if the file format is valid for the given formatValue
      if (!acceptedTypes.includes(currentFileExtension)) {
        const errorMessage =
          formatValue === "format2"
            ? "Please upload only Excel (.xlsx) files"
            : "Please upload only PDF, Excel (.xlsx), or Word (.docx) files";

        set_api_message([errorMessage, "orange"]);
        setTimeout(() => {
          set_api_message([]);
        }, 3000);

        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
        return;
      }

      const newFiles = Array.from(files);
      const uniqueFiles = newFiles.filter(
        (file) =>
          !storeFiles.some((existingFile) => existingFile.name === file.name)
      );

      // Merge new files with existing ones
      const updatedFiles = [...storeFiles, ...uniqueFiles];

      // Keep only the latest 5 files
      const limitedFiles = updatedFiles.slice(-5);

      // Update state with the limited files
      setStoreFiles(limitedFiles);
    }

    // Clear file input value
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // const storingFile = (event: React.ChangeEvent<HTMLInputElement>) => {
  //   const files = event.target.files;
  //   formatValue;
  //   if (files && files.length > 0) {
  //     const acceptedTypes = [".pdf", ".xlsx", ".docx"];

  //     // Get first file's format of current selection
  //     const currentFileExtension = `.${files[0].name
  //       .split(".")
  //       .pop()
  //       ?.toLowerCase()}`;

  //     // Check if first file is a valid format
  //     if (!acceptedTypes.includes(currentFileExtension)) {
  //       set_api_message([
  //         "Please upload only PDF, Excel (.xlsx), or Word (.docx) files",
  //         "orange",
  //       ]);
  //       setTimeout(() => {
  //         set_api_message([]);
  //       }, 3000);
  //       if (fileInputRef.current) {
  //         fileInputRef.current.value = "";
  //       }
  //       return;
  //     }

  //     const newFiles = Array.from(files);
  //     const uniqueFiles = newFiles.filter(
  //       (file) =>
  //         !storeFiles.some((existingFile) => existingFile.name === file.name)
  //     );

  //     // Merge new files with existing ones
  //     const updatedFiles = [...storeFiles, ...uniqueFiles];

  //     // Keep only the latest 5 files
  //     const limitedFiles = updatedFiles.slice(-5);

  //     // Update state with the limited files
  //     setStoreFiles(limitedFiles);
  //   }

  //   // Clear file input value
  //   if (fileInputRef.current) {
  //     fileInputRef.current.value = "";
  //   }
  // };

  useEffect(() => {
    sendingStoreMultipleFileToParent(storeFiles);
  }, [storeFiles]);

  const handleFileUpload = (files: FileList | null) => {
    if (files && files.length > 0) {
      const fileLists = Array.from(files);
      const fileNames = fileLists.map((file) => file.name);
      setIsSelectedFileName(fileNames);
    } else {
    }
  };

  const handleFileInputChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const files = event.target.files;
    handleFileUpload(files);
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
      setStoreFiles([]);
      resetColumn();
      setSelectedColumnValues({});
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }, [reset]);

  const removeFile = (name: string) => {
    setStoreFiles((prevFiles) =>
      prevFiles.filter((file) => file.name !== name)
    );
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };
  const [selectedColumnValues, setSelectedColumnValues] = useState<{
    [key: string]: string[];
  }>({});

  const handleColumnDropDownValueChange = (
    fileName: string,
    e: { value: { name: string }[] }
  ) => {
    const transformedSelection = e.value.map((item) => item.name);

    setSelectedColumnValues((prevState) => ({
      ...prevState,
      [fileName]: transformedSelection,
    }));
  };

  useEffect(() => {
    sendingSelectedMultipleColumnValues(selectedColumnValues);
  }, [selectedColumnValues]);

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
          onChange={(e) => {
            storingFile(e);
            // handleFileInputChange(e);
          }}
        />
      </div>

      {activateFileName && storeFiles?.length > 0 && (
        <div className="mfSelectedFileName">
          {/* <h3>Selected files:</h3> */}
          {storeFiles.map((fileName, index) => (
            <div key={index}>
              <div className="cont" onClick={() => {}}>
                <p className="mfSelectedFileNameSpan">
                  {getFileName(fileName?.name)}
                </p>
                <p
                  className="sendIcon"
                  style={{ cursor: "pointer" }}
                  onClick={() => {
                    removeFile(fileName?.name);
                    // delete storeExcelColumn["fileName?.name"];
                    removeObjectProperties(fileName?.name);
                  }}
                >
                  &times;
                </p>
              </div>
              {/* <div className="selectLayoutEx-cols">
                <label
                  htmlFor="selectLayoutExcelColumn"
                  style={{ marginLeft: "10px" }}
                >
                  Select layout Excel column
                </label>
                <MultiSelect
                  value={storeExcelColumn[fileName?.name]
                    ?.filter((column) =>
                      selectedColumnValues[fileName?.name]?.includes(column)
                    )
                    .map((value) => ({ name: value }))}
                  onChange={(e) =>
                    handleColumnDropDownValueChange(fileName?.name, e)
                  }
                  options={
                    storeExcelColumn[fileName?.name]
                      ?.filter(
                        (value: string) => !excludedColumn.includes(value)
                      )
                      .map((value: string) => ({ name: value })) || []
                  }
                  optionLabel="name"
                  placeholder="Select Excel column"
                  maxSelectedLabels={2}
                  style={{ maxWidth: "200px" }}
                />
              </div> */}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default FacilitiesBrowseMultipleFile;
