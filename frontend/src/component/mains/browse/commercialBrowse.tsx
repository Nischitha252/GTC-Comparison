import React, { useEffect, useRef, useState } from "react";
import "./commercialBrowse.css";
import UploadIcon from "assets/icon/ip-upload-icon";
import * as XLSX from "xlsx";
import { MultiSelect } from "primereact/multiselect";
import "primereact/resources/themes/saga-blue/theme.css";
import "primereact/resources/primereact.min.css";
import "primeicons/primeicons.css";
import CommercialGetColumn from "service/getColumn";

interface BrowseFileProps {
  inputId: string;
  multiple: boolean;
  disabled: boolean;
  reset?: boolean;
  activateFileName?: boolean;
  sendingStoreFileToParent: React.Dispatch<React.SetStateAction<File[]>>;
  sendingSelectedColumnDropDownValues: (values: string[]) => void;
  set_api_message: (message: string[]) => void;
}

const CommercialBrowseFile: React.FC<BrowseFileProps> = ({
  inputId,
  multiple,
  disabled,
  reset,
  activateFileName,
  sendingStoreFileToParent,
  sendingSelectedColumnDropDownValues,
  set_api_message,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isStoreFile, setStoreFile] = useState<File[]>([]);
  const [isExcelExtractColumns, setIsExcelExtractColumns] = useState<string[]>(
    []
  );
  const [selectedColumnValues, setSelectedColumnValues] = useState<string[]>(
    []
  );
  const [defaultSelectedValuess, setDefaultSelectedValues] = useState<string[]>(
    []
  );
  const [excludedColumns, setExcludedColumns] = useState<string[]>([]);
  // Default column values to select
  const defaultSelectedValues = ["Description", "Qty", "UOM", "Manufacturer"];
  let arr;

  const getColumnApi = async () => {
    try {
      const response = await CommercialGetColumn(isStoreFile, set_api_message);

      if (response) {
        arr = response?.data?.column_mapping;

        setDefaultSelectedValues(Object.values(arr));
        setExcludedColumns(Object.keys(arr));
      }
    } catch (error) {
      console.error("Error uploading or processing files:", error);
    }
    //
  };

  useEffect(() => {
    const validDefaults = defaultSelectedValuess?.filter((value) =>
      isExcelExtractColumns.includes(value)
    );
    if (validDefaults) {
      setSelectedColumnValues(validDefaults);

      sendingSelectedColumnDropDownValues(validDefaults);
    }
  }, [defaultSelectedValuess]);

  const handleFileInputChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (file) {
      const acceptedTypes = [".xlsx"];

      const fileExtension = `.${file.name.split(".").pop()?.toLowerCase()}`;

      if (!acceptedTypes.includes(fileExtension)) {
        // Invalid file type
        set_api_message(["Upload File is not a valid format", "orange"]);
        setTimeout(() => {
          set_api_message([]);
        }, 3000);
        // Reset the file input
        event.target.value = "";
        return;
      }
      storingFile(event);
      const reader = new FileReader();
      reader.onload = (e) => {
        const data = new Uint8Array(e.target!.result as ArrayBuffer);
        const workbook = XLSX.read(data, { type: "array" });
        const sheetName = workbook.SheetNames[0]; // Get the first sheet
        const sheet = workbook.Sheets[sheetName];
        const range = XLSX.utils.decode_range(sheet["!ref"]!); // Get the range of cells

        const columns: string[] = [];
        for (let col = range.s.c; col <= range.e.c; col++) {
          const cellAddress = { c: col, r: 0 }; // Get the first row
          const cellRef = XLSX.utils.encode_cell(cellAddress);
          const cell = sheet[cellRef];
          if (cell) {
            columns.push(cell.v as string);
          }
        }
        setIsExcelExtractColumns(columns);

        // Set default selected values if they exist in the extracted columns
        const validDefaults = defaultSelectedValues.filter((value) =>
          columns.includes(value)
        );
        setSelectedColumnValues(validDefaults);

        sendingSelectedColumnDropDownValues(validDefaults);
      };
      reader.readAsArrayBuffer(file);
    }
  };
  useEffect(() => {
    if (isStoreFile.length > 0) getColumnApi();
  }, [isStoreFile]);

  const storingFile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      const newFiles = Array.from(files);
      const uniqueFiles = newFiles.filter(
        (file) =>
          !isStoreFile.some((existingFile) => existingFile.name === file.name)
      );
      setStoreFile(uniqueFiles);
      sendingStoreFileToParent(uniqueFiles);
    }
  };

  useEffect(() => {
    if (reset) {
      setSelectedColumnValues([]);
      setStoreFile([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }, [reset]);

  const columns = isExcelExtractColumns.map((value) => ({ name: value }));

  return (
    <div style={{ marginTop: "-15px" }}>
      <div
        style={{
          display: "flex",
          width: "100%",
          justifyContent: "space-between",
        }}
      >
        <div className="browseBox">
          <label
            htmlFor={inputId}
            className={`browseFile ${disabled ? "disabled" : ""}`}
          >
            <UploadIcon
              className={`abbIcon ${disabled ? "disabled" : ""}`}
              size="large"
            />
            <span>Browse files</span>
          </label>

          <input
            type="file"
            id={inputId}
            accept=".xlsx"
            multiple={multiple}
            disabled={disabled}
            style={{ display: "none" }}
            onChange={handleFileInputChange}
            ref={fileInputRef}
          />
        </div>

        {isStoreFile.length === 1 && defaultSelectedValuess.length < 1 ? (
          <div className="shimmer-container animate-pulse">
            <div className="shimmer"></div>
          </div>
        ) : (
          defaultSelectedValuess?.length > 0 && (
            <div className="selectLayoutEx-col">
              <label
                htmlFor="selectLayoutExcelColumn"
                className="comBLabelHeading"
              >
                Select layout Excel column
              </label>
              <MultiSelect
                value={columns.filter((column) =>
                  selectedColumnValues.includes(column.name)
                )}
                onChange={(e) => {
                  const transformedSelection = e.value.map(
                    (item: any) => item.name
                  );
                  setSelectedColumnValues(transformedSelection);
                  sendingSelectedColumnDropDownValues(transformedSelection);
                }}
                options={columns}
                optionLabel="name"
                placeholder="Select Excel column"
                maxSelectedLabels={2}
                style={{ width: "200px" }}
              />
            </div>
          )
        )}
      </div>

      {activateFileName && isStoreFile.length > 0 && (
        <div className="selectedFileNameP">
          {isStoreFile.map((fileName, index) => (
            <span key={index} className="selectedFileNameChildP">
              {fileName.name}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

export default CommercialBrowseFile;
