// multiple.tsx
import React, { useEffect, useRef, useState } from "react";
import "./commercialComp.css";
import BrowseFile from "../browse/browse";
import CommercialPostFile from "service/commercialMaterialProcess";
import Loader from "../loader/loader";
import ProcessFile from "service/processFile";
import LeftArrowIcon from "assets/icon/ip-leftArrow-icon";
import CommercialBrowseFile from "../browse/commercialBrowse";
import CommercialBrowseMultipleFile from "../browse/commercialMultipleBrowse";
import CommercialComparison from "../../pages/comparison/commercialComparison";
import DownloadIcon from "assets/icon/ip-download-icon";
import { HandleFileUploadMultipleExcelExtraction } from "../../../utils/helperFunction";
import FacilitateComp from "./facilitateComp";
import WelcomeText from "component/mains/welcomeText/welcomeText";
import { Dropdown, DropdownChangeEvent } from "primereact/dropdown";
import "primereact/resources/themes/saga-blue/theme.css";
import "primereact/resources/primereact.min.css";
import "primeicons/primeicons.css";
import Notification from "../notification/notification";

type DataList = {
  [entity: string]: {
    similarities: string[];
    additions: string[];
    removals: string[];
    differences: string;
  };
};
interface RFQ {
  name: string;
  value: string;
}
type RadioValue = "with_format" | "without_format";

interface CommercialCompProps {
  onBackClick: () => void;
  onUploadClick: () => void;
  setIsLoader: (isLoading: boolean) => void;
  onDataListUpdate: (dataList: DataList) => void;
  onDownloadClick: (blobName: string) => void;
  onTokenCost: (tokenCost: number) => void;
}

const CommercialComp: React.FC<CommercialCompProps> = ({
  onBackClick,
  onUploadClick,
  setIsLoader,
}) => {
  const [selectedPreloadAbbGTC, setSelectedPreloadAbbGTC] =
    useState<string>("");
  const [abbGtcFileUpload, setAbbGtcFileUpload] = useState<FileList | null>(
    null
  );
  const [supplierFileUploads, setSupplierFileUploads] = useState<
    (FileList | null)[]
  >([]);
  const [numberOfSuppliers, setNumberOfSuppliers] = useState<number>(0);
  const [isReset, setIsReset] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const scrollBottomRef = useRef<HTMLDivElement>(null);
  const [isStoreFile, setStoreFile] = useState<File[]>([]);
  const [selectedColumnValues, setSelectedColumnValues] = useState<string[]>(
    []
  );
  const [multipleStoreFiles, setMultipleStoreFiles] = useState<File[]>([]);
  const [selectedMultipleColumnValues, setSelectedMultipleColumnValues] =
    useState<{
      [key: string]: string[];
    }>({});
  const handleAbbGTCChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedPreloadAbbGTC(event.target.value);
  };
  // const [isApiMessage, setApiMessage] = useState<boolean>(false);
  const [tableData, setTableData] = useState<any>([]);
  const [renderBackboolean, setRenderBackboolean] = useState<boolean>(false);
  const [selectedValue, setSelectedValue] = useState<string>("");
  const [data, setRawData] = useState<any[]>([]);
  const [processedData, setProcessedData] = useState<any[]>([]);
  const [originalData, setOriginalData] = useState<any[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [facilitateData, setFacilitateData] = useState<{}>({});
  const [api_Message, setApiMessage] = useState<string[]>([]);
  const { exportToCSV, storeExcelColumn } =
    HandleFileUploadMultipleExcelExtraction();

  const handleUpload = async () => {
    setIsLoading(true);
    try {
      const response = await CommercialPostFile(
        isStoreFile,
        selectedColumnValues,
        multipleStoreFiles,
        selectedMultipleColumnValues,
        selectedValue,
        setApiMessage
      );

      if (response) {
        // setApiMessage(true);
        setTableData(response?.result);
        setIsLoading(false);
        setRenderBackboolean(true);
        setRawData(response);
        // onUploadClick();
      }
      // setTimeout(() => {
      //   setApiMessage(false);
      // }, 3000);
    } catch (error) {
      console.error("Error uploading or processing files:", error);
      setIsLoading(false);
    }
  };

  const cities: RFQ[] = [
    {
      name: "Facilities",
      value: "Facilities",
    },
    {
      name: "Material",
      value: "Material",
    },
  ];

  const handleReset = () => {
    setIsReset(true);
    setSelectedPreloadAbbGTC("");
    setAbbGtcFileUpload(null);
    setSupplierFileUploads([]);

    setNumberOfSuppliers(0);
    setTimeout(() => {
      setIsReset(false);
    }, 500);
  };
  const isSupplierUploadEnabled =
    selectedPreloadAbbGTC &&
    (selectedPreloadAbbGTC !== "others" || abbGtcFileUpload);
  const areSupplierGtcFilesUploaded = supplierFileUploads.every(
    (file) => file !== null
  );
  // Handler function for radio button change
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value as RadioValue; // Type assertion
    setSelectedValue(value); // This logs the selected value to the console
  };
  // Call scrollToBottom whenever content changes
  useEffect(() => {
    scrollBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [
    isSupplierUploadEnabled,
    numberOfSuppliers,
    areSupplierGtcFilesUploaded,
    isReset,
  ]);

  useEffect(() => {
    setRenderBackboolean(false);
  }, []);

  useEffect(() => {
    if (tableData && typeof tableData === "object") {
      const dataHeaders = Object.keys(tableData);
      setHeaders(dataHeaders);

      const rows = dataHeaders[0]
        ? tableData[dataHeaders[0]].map((_: any, index: number) => {
            const row: any = {};
            dataHeaders.forEach((header) => {
              row[header] = tableData[header][index];
            });
            return row;
          })
        : [];

      setProcessedData(rows);
      setOriginalData(rows); // Store the original data
    }
  }, [tableData]);

  const filterEmptyRows = () => {
    // Get all columns that contain .xlsx in their name
    const columnsToCheck = headers.filter((header) => header.includes(".xlsx"));

    if (columnsToCheck.length === 0) {
      console.warn("No columns found with .xlsx in the name");
      return;
    }

    const filteredRows = processedData.filter((row) => {
      // Check each column in the row
      return columnsToCheck.every((column) => {
        const value = row[column];

        // Handle different types of empty values
        if (value === "" || value === null || value === undefined) {
          return false;
        }

        const stringValue = String(value).trim();
        if (
          stringValue === "" ||
          stringValue === "-" ||
          stringValue === "N/A"
        ) {
          return false;
        }

        const numValue = parseFloat(stringValue);
        if (!isNaN(numValue) && numValue === 0) {
          return false;
        }

        return true;
      });
    });

    // Log filtering results

    // If all rows would be filtered out, show warning and don't update
    if (filteredRows.length === 0) {
      console.warn("Filtering would remove all rows - operation cancelled");
      return;
    }

    // Update the processed data with filtered rows
    setProcessedData(filteredRows);
  };

  const restoreOriginalData = () => {
    setProcessedData(originalData);
  };

  return (
    <>
      {tableData.length === 0 &&
        Object.values(facilitateData || {}).length === 0 && <WelcomeText />}

      {api_Message.length > 0 && (
        <Notification message={api_Message[0]} color={api_Message[1]} />
      )}
      {tableData.length >= 0 &&
        Object.keys(facilitateData).length < 1 &&
        !isLoading && (
          <button className="comMultipleBackButton" onClick={onBackClick}>
            <LeftArrowIcon
              className="comMultipleBackIcon"
              // name="abb/left-arrow"
              size="medium"
            />
            Back
          </button>
        )}
      {tableData.length >= 0 && (
        <div
          className="comMultipleGTC"
          style={{
            width: Object.keys(facilitateData).length > 1 ? "100%" : "",
          }}
        >
          {!isLoading && (
            <>
              {Object.keys(facilitateData).length < 1 && (
                <>
                  <div className="comPreloadAndUserUpload">
                    <label
                      htmlFor="selectPreloadAbbGTC"
                      className="comLabelHeading"
                      style={{ marginRight: "80px" }}
                    >
                      Choose Category:
                    </label>

                    <Dropdown
                      value={selectedPreloadAbbGTC}
                      onChange={(e: DropdownChangeEvent) => {
                        setSelectedPreloadAbbGTC(e.value);
                      }}
                      options={cities}
                      optionLabel="name"
                      placeholder="Select a category"
                      className="rfq-dropdown"
                    />
                  </div>
                </>
              )}

              {selectedPreloadAbbGTC === "Facilities" && (
                <FacilitateComp
                  onBackClick={onBackClick}
                  setFacilitateData={setFacilitateData}
                  set_Api_message={setApiMessage}
                  setSelectedPreloadAbbGTC={setSelectedPreloadAbbGTC}
                />
              )}
              {selectedPreloadAbbGTC === "Material" &&
                Object.keys(facilitateData).length < 1 && (
                  <>
                    <div className="radio-container">
                      <label>
                        <input
                          type="radio"
                          name="format"
                          value="with_format"
                          checked={selectedValue === "with_format"}
                          onChange={handleChange}
                        />
                        With Format
                      </label>
                      <label>
                        <input
                          type="radio"
                          name="format"
                          value="without_format"
                          checked={selectedValue === "without_format"}
                          onChange={handleChange}
                        />
                        Without Format
                      </label>
                    </div>

                    {selectedValue === "with_format" && (
                      <>
                        <div className="comSupplierFileUploadsTextExcel comLabelHeading">
                          Upload supplier(s) Layout:
                        </div>
                        <div className="comSupplierFileUploadsBox">
                          {Array.from({ length: 1 }).map((_, index) => (
                            <div key={index} className="comSupplierFileUpload">
                              <CommercialBrowseFile
                                inputId={`supplierFileUpload_${index}`}
                                multiple={false}
                                disabled={
                                  !selectedPreloadAbbGTC ||
                                  !isSupplierUploadEnabled
                                }
                                reset={isReset}
                                activateFileName={true}
                                sendingStoreFileToParent={setStoreFile}
                                sendingSelectedColumnDropDownValues={
                                  setSelectedColumnValues
                                }
                                set_api_message={setApiMessage}
                              />
                            </div>
                          ))}
                        </div>
                      </>
                    )}

                    <div
                      className="comSupplierFileUploadsText comLabelHeading"
                      style={{ marginTop: "60px" }}
                    >
                      Upload supplier(s) Commercial:
                      <br />
                      <span style={{ fontWeight: "700" }}>
                        Max 5 files (xlsx)
                      </span>
                    </div>

                    <div className="supplierFileUploadsBox">
                      {Array.from({ length: 1 }).map((_, index) => (
                        <div key={index} className="comSupplierFileUpload">
                          <CommercialBrowseMultipleFile
                            inputId={`supplierMultipleFileUpload_${index}`}
                            disabled={
                              !selectedPreloadAbbGTC || !isSupplierUploadEnabled
                            }
                            reset={isReset}
                            activateFileName={true}
                            accept=".xlsx"
                            sendingStoreMultipleFileToParent={
                              setMultipleStoreFiles
                            }
                            sendingSelectedMultipleColumnValues={
                              setSelectedMultipleColumnValues
                            }
                            excludedColumn={selectedColumnValues}
                            api_message={setApiMessage}
                          />
                        </div>
                      ))}
                    </div>
                    <div className="comResetAndUploadButtonBox">
                      <button className="comResetButton" onClick={handleReset}>
                        Reset
                      </button>

                      <button
                        className="comUploadButton"
                        onClick={handleUpload}
                        disabled={
                          (selectedValue == "with_format" &&
                            isStoreFile.length < 1) ||
                          multipleStoreFiles.length < 1
                        }
                      >
                        Upload
                      </button>
                    </div>
                  </>
                )}
            </>
          )}

          <div ref={scrollBottomRef}></div>
        </div>
      )}
      {isLoading && <Loader loaderContent="Analyzing..." />}
      {renderBackboolean && (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            width: "100%",
          }}
        >
          <div>
            <button className="multipleBackButton" onClick={onBackClick}>
              <LeftArrowIcon className="multipleBackIcon" size="medium" />
              Back
            </button>
          </div>
          <div style={{ display: "flex", gap: "15px" }}>
            <div>
              <button
                className="downloadButton"
                onClick={() => {
                  filterEmptyRows();
                }}
              >
                Filter by Price
              </button>
            </div>
            <div>
              <button
                className="downloadButton"
                onClick={() => {
                  restoreOriginalData();
                }}
              >
                Exclude Price
              </button>
            </div>
          </div>
          <div>
            <button
              className="downloadButton"
              onClick={() => {
                exportToCSV(processedData, headers, "commercial_comparison");
              }}
            >
              <DownloadIcon className="downloadBackIcon" size="medium" />
              Download
            </button>
          </div>
        </div>
      )}

      {tableData && (
        <CommercialComparison headers={headers} data={processedData} />
      )}
    </>
  );
};

export default CommercialComp;
