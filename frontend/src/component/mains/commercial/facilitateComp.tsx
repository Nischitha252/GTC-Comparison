// multiple.tsx
import React, { useEffect, useRef, useState } from "react";
import "./facilitateComp.css";
import BrowseFile from "../browse/browse";
import FacilitiesLayoutPostFile from "service/facilitiesLayoutFileProcess";
import Loader from "../loader/loader";
import ProcessFile from "service/processFile";
import LeftArrowIcon from "assets/icon/ip-leftArrow-icon";
import FacilitiesBrowseFile from "../browse/facilitiesBrowse";
import FacilitiesBrowseMultipleFile from "../browse/facilitiesMultipleBrowse";
import FacilitateComparison from "../../pages/comparison/facilitateComparison";
import DownloadIcon from "assets/icon/ip-download-icon";
import { HandleFileUploadMultipleExcelExtraction } from "../../../utils/helperFunction";
import FacilitiesMultiplePostFile from "service/facilitiesMultipleFileProces";
import InfoIcon from "assets/icon/info";
import format1 from "../../../assets/icon/format1.png";
import format2 from "../../../assets/icon/format2.png";

// type DataList = {
//   [entity: string]: {
//     similarities: string[];
//     additions: string[];
//     removals: string[];
//     Difference: string;
//   };
// };

type RadioValue = "with_format" | "without_format";

interface FacilitateCompProps {
  onBackClick: () => void;
  setFacilitateData: (value: any) => void;
  setIsLoader?: (isLoading: boolean) => void;
  // onDataListUpdate: (dataList: DataList) => void;
  // onDownloadClick: (blobName: string) => void;
  // onTokenCost: (tokenCost: number) => void;
  set_Api_message: (message: string[]) => void;
  setSelectedPreloadAbbGTC: (value: string) => void;
}

const FacilitateComp: React.FC<FacilitateCompProps> = ({
  onBackClick,
  setFacilitateData,
  setIsLoader,
  set_Api_message,
  setSelectedPreloadAbbGTC,
}) => {
  // const [selectedPreloadAbbGTC, setSelectedPreloadAbbGTC] =
  //   useState<string>("");
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
  // const handleAbbGTCChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
  //   setSelectedPreloadAbbGTC(event.target.value);
  // };
  const [isApiMessage, setApiMessage] = useState<boolean>(false);
  const [tableData, setTableData] = useState<any>([]);
  const [renderBackboolean, setRenderBackboolean] = useState<boolean>(false);
  const [selectedValue, setSelectedValue] = useState<string>("format1");
  const { exportToCSVFacilitate, exportFacilitateToXLSX } =
    HandleFileUploadMultipleExcelExtraction();
  const [zoomedImage, setZoomedImage] = useState<string | null>(null);

  const handleImageClick = (src: string) => {
    setZoomedImage(src);
  };

  const closeZoom = () => {
    setZoomedImage(null);
  };

  const makeAPi = async () => {
    try {
      const response = await FacilitiesMultiplePostFile(
        isStoreFile,
        multipleStoreFiles,
        set_Api_message,
        selectedValue
      );
      if (response) {
        setApiMessage(true);
        setTableData(response?.data);
        setFacilitateData(response?.data);
        setIsLoading(false);
        setRenderBackboolean(true);
        // setRawData(response);
      }
      setTimeout(() => {
        setApiMessage(false);
      }, 3000);
    } catch (error) {
      console.error("Error uploading or processing files:", error);
      setIsLoading(false);
    }
  };

  const handleUpload = () => {
    setIsLoading(true);
    makeAPi();
  };
  let parsedTableData;
  try {
    parsedTableData =
      typeof tableData === "string"
        ? JSON.parse(tableData.replace(/NaN/g, "null")) // Replace NaN with null
        : tableData;
  } catch (error) {
    console.error("Error parsing tableData:", error);
    parsedTableData = null;
  }
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
  // const isSupplierUploadEnabled =
  //   selectedPreloadAbbGTC &&
  //   (selectedPreloadAbbGTC !== "others" || abbGtcFileUpload);
  // const areSupplierGtcFilesUploaded = supplierFileUploads.every(
  //   (file) => file !== null
  // );
  // Handler function for radio button change
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value as RadioValue; // Type assertion
    setSelectedValue(value); // This logs the selected value to the console
  };
  // Call scrollToBottom whenever content changes
  useEffect(() => {
    scrollBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [
    // isSupplierUploadEnabled,
    numberOfSuppliers,
    // areSupplierGtcFilesUploaded,
    isReset,
  ]);

  useEffect(() => {
    setRenderBackboolean(false);
  }, []);

  return (
    <>
      {!isLoading && Object.keys(tableData).length < 1 && (
        <>
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              marginTop: "-30px",
            }}
            className="radio-container"
          >
            <label>
              <input
                type="radio"
                name="format"
                value="format1"
                checked={selectedValue === "format1"}
                onChange={handleChange}
                style={{
                  cursor: "pointer",
                }}
              />
              Format1
            </label>
            <br />
            <label>
              <input
                style={{
                  cursor: "pointer",
                }}
                type="radio"
                name="format"
                value="format2"
                checked={selectedValue === "format2"}
                onChange={handleChange}
              />
              Format2
            </label>
            <br />
            <div className="tooltip-container">
              <InfoIcon size="medium" className="info" />
              <div className="tooltip">
                <div className="tooltip-header">Format1 info</div>
                <div className="tooltip-body">
                  <strong>Format 1</strong> is a highly flexible solution
                  designed to handle files with mismatched structures or
                  incomplete data. It supports multiple file types, including
                  PDF, Word, and Excel, and allows users to specify only the
                  required columns for comparison. This adaptability makes it
                  ideal for complex or irregular document structures.
                  Additionally, Format 1 can seamlessly handle documents that
                  align with the structured requirements of Format 2, offering a
                  comprehensive and inclusive comparison capability.
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "center",
                      marginTop: "8px",
                      borderRadius: "16px",
                    }}
                  >
                    <img
                      src={format1}
                      width="300px"
                      height="200px"
                      className="zoomable"
                      alt="Format 1 example"
                      onClick={() => handleImageClick(format1)}
                    />
                  </div>
                </div>
                <div className="tooltip-header">Format2 info</div>
                <div className="tooltip-body">
                  <strong>Format 2</strong> is tailored specifically for Excel
                  files, prioritizing precision and speed. It requires fully
                  populated columns with headers positioned at the top that
                  match the reference format exactly. Additional columns can
                  also be incorporated for comparison with other documents. This
                  structured and consistent approach ensures exceptional
                  accuracy and efficiency, making it ideal for well-organized
                  data.
                  <br />
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "center",
                      marginTop: "8px",
                      borderRadius: "16px",
                    }}
                  >
                    <img
                      src={format2}
                      width="300px"
                      height="200px"
                      className="zoomable"
                      alt="Format 2 example"
                      onClick={() => handleImageClick(format2)}
                    />
                  </div>
                </div>
              </div>

              {zoomedImage && (
                <div className="image-zoom-modal" onClick={closeZoom}>
                  <img
                    src={zoomedImage}
                    alt="Zoomed"
                    className="zoomed-image"
                  />
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {!isLoading && Object.keys(tableData).length < 1 && (
        <div className="facMultipleGTC">
          <>
            <div className="facSupplierFileUploadsTextExcel facLabelHeading">
              Upload supplier(s) Layout:
            </div>
            <div className="facSupplierFileUploadsBox">
              {Array.from({ length: 1 }).map((_, index) => (
                <div key={index} className="facSupplierFileUpload">
                  <FacilitiesBrowseFile
                    inputId={`supplierFileUpload_${index}`}
                    multiple={false}
                    // disabled={!selectedPreloadAbbGTC || !isSupplierUploadEnabled}
                    reset={isReset}
                    activateFileName={true}
                    sendingStoreFileToParent={setStoreFile}
                    sendingSelectedColumnDropDownValues={
                      setSelectedColumnValues
                    }
                    set_api_message={set_Api_message}
                  />
                </div>
              ))}
            </div>
          </>

          <div className="facSupplierFileUploadsText facLabelHeading">
            Upload supplier(s) Facilities:
            <br />
            <span style={{ fontWeight: "700" }}>
              {selectedValue === "format1"
                ? "A maximum of 5 files (either PDF or XLSX or docx)"
                : "A maximum of 5 files (either XLSX)"}
            </span>
          </div>

          <div className="facSupplierFileUploadsBox">
            {Array.from({ length: 1 }).map((_, index) => (
              <div key={index} className="facSupplierFileUpload">
                <FacilitiesBrowseMultipleFile
                  inputId={`supplierMultipleFileUpload_${index}`}
                  // multiple={false}
                  // disabled={!selectedPreloadAbbGTC || !isSupplierUploadEnabled}
                  reset={isReset}
                  activateFileName={true}
                  accept={
                    selectedValue == "format1" ? ".pdf,.xlsx,.docx" : ".xlsx"
                  }
                  sendingStoreMultipleFileToParent={setMultipleStoreFiles}
                  sendingSelectedMultipleColumnValues={
                    setSelectedMultipleColumnValues
                  }
                  excludedColumn={selectedColumnValues}
                  set_api_message={set_Api_message}
                  formatValue={selectedValue}
                />
              </div>
            ))}
          </div>

          <div className="facResetAndUploadButtonBox">
            <button className="facResetButton" onClick={handleReset}>
              Reset
            </button>

            <button
              className="facUploadButton"
              onClick={handleUpload}
              disabled={isStoreFile.length < 1 || multipleStoreFiles.length < 1}
            >
              Upload
            </button>
          </div>

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
            <button className="facMultipleBackButton" onClick={onBackClick}>
              <LeftArrowIcon className="facMultipleBackIcon" size="medium" />
              Back
            </button>
          </div>
          <div>
            <button
              className="downloadButton"
              onClick={() => {
                if (parsedTableData?.data) {
                  exportFacilitateToXLSX(parsedTableData?.data);
                } else {
                  exportToCSVFacilitate(tableData);
                }
              }}
            >
              <DownloadIcon className="downloadBackIcon" size="medium" />
              Download
            </button>
          </div>
        </div>
      )}

      {tableData && <FacilitateComparison tableData={tableData} />}
    </>
  );
};

export default FacilitateComp;
