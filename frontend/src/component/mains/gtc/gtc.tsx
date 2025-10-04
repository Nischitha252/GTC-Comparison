// multiple.tsx
import React, { useEffect, useRef, useState } from "react";
import "./gtc.css";
import BrowseFile from "../browse/browse";
import PostFile from "service/postFile";
import Loader from "../loader/loader";
import ProcessFile from "service/processFile";
import LeftArrowIcon from "assets/icon/ip-leftArrow-icon";
import { useLocation } from "react-router-dom";
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

interface GTC {
  name: string;
  value: string;
}

interface GTCProps {
  onBackClick: () => void;
  onUploadClick: () => void;
  setIsLoader: (isLoading: boolean) => void;
  onDataListUpdate: (dataList: DataList) => void;
  onDownloadClick: (blobName: string) => void;
  onTokenCost: (tokenCost: number) => void;
}

const GTC: React.FC<GTCProps> = ({
  onBackClick,
  onUploadClick,
  setIsLoader,
  onDataListUpdate,
  onDownloadClick,
  onTokenCost,
}) => {
  const [selectedPreloadAbbGTC, setSelectedPreloadAbbGTC] =
    useState<string>("");
  const [isIndexName, setIsIndexName] = useState<string | null>(null);
  const [abbGtcFileUpload, setAbbGtcFileUpload] = useState<FileList | null>(
    null
  );
  const location = useLocation();
  const currentUrl = `${window.location.origin}${location.pathname}${location.search}`;
  const cities: GTC[] = [
    {
      name: "ABB GTC Goods and Services (2020-2 Standard)",
      value: "ABB GTC Goods and Services (2020-2 Standard)",
    },
    {
      name: "ABB GTC Goods and Services (2021 Italy)",
      value: "ABB GTC Goods and Services (2021 Italy)",
    },
    {
      name: "ABB GTC IT Procurement (2023-04)",
      value: "ABB GTC IT Procurement (2023-04)",
    },
    {
      name: "ABB GTC IT Procurement Hardware Schedule (2022-03)",
      value: "ABB GTC IT Procurement Hardware Schedule (2022-03)",
    },
    {
      name: "ABB GTC IT Procurement Software License Schedule (2023-04)",
      value: "ABB GTC IT Procurement Software License Schedule (2023-04)",
    },
    {
      name: "ABB GTC Goods and Services (2024-1 STANDARD)",
      value: "ABB GTC Goods and Services (2024-1 STANDARD)",
    },
    { name: "Others (Upload your GTC)", value: " " },
  ];

  const [supplierFileUploads, setSupplierFileUploads] = useState<
    (FileList | null)[]
  >([]);
  const [numberOfSuppliers, setNumberOfSuppliers] = useState<number>(0);
  const [isReset, setIsReset] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const scrollBottomRef = useRef<HTMLDivElement>(null);
  const [isTokenCost, setIsTokenCost] = useState<number>(0);
  const [api_Message, setApiMessage] = useState<string[]>([]);

  const handleAbbGTCChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedPreloadAbbGTC(event.target.value);
    setAbbGtcFileUpload(null);
  };

  const handleUpload = async () => {
    // Implement your upload logic here
    setIsLoading(true);
    setIsLoader(false);

    const supplierFile = supplierFileUploads[0];
    let abbGtcFile = null;
    if (abbGtcFileUpload) {
      abbGtcFile = abbGtcFileUpload[0];
    }

    let isIndexName = null;
    if (selectedPreloadAbbGTC !== " ") {
      isIndexName = selectedPreloadAbbGTC;
    }

    if (supplierFile && supplierFile.length > 0) {
      try {
        const reponse = await PostFile(
          isIndexName,
          abbGtcFile,
          supplierFile[0],
          setApiMessage
        );

        const processResponse = await ProcessFile(
          isIndexName,
          setApiMessage,
          reponse
        );
        const dataList = processResponse.result;
        const blobName = processResponse.blob_name;
        const tokenCost = processResponse.total_cost;
        onDataListUpdate(dataList);
        onDownloadClick(blobName);
        onTokenCost(tokenCost);
        onUploadClick();

        setIsLoading(false);
        setIsLoader(false);
      } catch (error) {
        console.error("Error uploading or processing files:", error);
        setIsLoading(false);
        setIsLoader(false);
      }
    }
    //
    // setIsLoading(false);
  };

  const handleAbbGtcFileUpload = (file: FileList | null) => {
    if (file && file.length > 0) {
      setAbbGtcFileUpload(file);
    }
  };

  const handleSupplierFileUpload = (index: number, file: FileList | null) => {
    const newUploads = [...supplierFileUploads];
    newUploads[index] = file;
    setSupplierFileUploads(newUploads);
  };

  const supplierOptions = Array.from({ length: 1 }, (_, i) => ({
    label: (i + 1).toString(),
    value: i + 1,
  }));

  const handleReset = () => {
    setIsReset(true);
    setSelectedPreloadAbbGTC("");
    setAbbGtcFileUpload(null);
    setSupplierFileUploads([]);
    setNumberOfSuppliers(0);

    setTimeout(() => {
      setIsReset(false);
    }, 0);
  };
  const isSupplierUploadEnabled =
    selectedPreloadAbbGTC &&
    (selectedPreloadAbbGTC !== "others" || abbGtcFileUpload);
  const areSupplierGtcFilesUploaded = supplierFileUploads.every(
    (file) => file !== null
  );

  useEffect(() => {
    scrollBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [
    isSupplierUploadEnabled,
    numberOfSuppliers,
    areSupplierGtcFilesUploaded,
    isReset,
  ]);
  const handleNumberOfSuppliersChange = (event: DropdownChangeEvent) => {
    const count = event.value as number; // Accessing the selected value
    setNumberOfSuppliers(count);
    setSupplierFileUploads(Array(count).fill(null));
  };

  const [isCentered, setIsCentered] = useState(false);

  useEffect(() => {
    // Function to check if the window width is between 950px and 1000px
    function updateDropdownPosition() {
      setIsCentered(window.innerWidth < 1800);
    }

    // Initial check
    updateDropdownPosition();

    // Update on resize
    window.addEventListener("resize", updateDropdownPosition);

    // Cleanup on unmount
    return () => window.removeEventListener("resize", updateDropdownPosition);
  }, []);

  return (
    <>
      {api_Message.length > 0 && (
        <Notification message={api_Message[0]} color={api_Message[1]} />
      )}

      <div className="multipleGTC">
        {!isLoading && (
          <>
            <button className="multipleBackButton" onClick={onBackClick}>
              <LeftArrowIcon
                className="multipleBackIcon"
                // name="abb/left-arrow"
                size="medium"
              />
              Back
            </button>
            <div className="main">
              <div>
                <div className="preloadAndUserUpload">
                  <label htmlFor="selectPreloadAbbGTC" className="labelHeading">
                    Select ABB GTC:{" "}
                  </label>
                  <Dropdown
                    value={selectedPreloadAbbGTC}
                    onChange={(e: DropdownChangeEvent) => {
                      setSelectedPreloadAbbGTC(e.value);
                    }}
                    options={cities}
                    optionLabel="name"
                    placeholder="Select a GTC"
                    className="dropdown"
                    style={{ width: "300px" }}
                    panelClassName={
                      isCentered ? "centered-dropdown" : "default-dropdown"
                    }
                  />
                  {/* <select
                    id="selectPreloadAbbGTC"
                    className="selectPreloadAbbGTC"
                    value={selectedPreloadAbbGTC}
                    onChange={handleAbbGTCChange}
                  >
                    <option />
                    <option value="ABB GTC Goods and Services (2020-2 Standard)">
                      ABB GTC Goods and Services (2020-2 Standard)
                    </option>
                    <option value="ABB GTC Goods and Services (2021 Italy)">
                      ABB GTC Goods and Services (2021 Italy)
                    </option>
                    <option value="ABB GTC IT Procurement (2023-04)">
                      ABB GTC IT Procurement (2023-04)
                    </option>
                    <option value="ABB GTC IT Procurement Hardware Schedule (2022-03)">
                      ABB GTC IT Procurement Hardware Schedule (2022-03)
                    </option>
                    <option value="ABB GTC IT Procurement Software License Schedule (2023-04)">
                      ABB GTC IT Procurement Software License Schedule (2023-04)
                    </option>
                    <option value="ABB GTC Goods and Services (2024-1 STANDARD)">
                      ABB GTC Goods and Services (2024-1 STANDARD)
                    </option>
                    <option value=" ">Others (Upload your GTC)</option>
                  </select> */}

                  {selectedPreloadAbbGTC === " " && (
                    <BrowseFile
                      inputId="AbbGtcFileUpload"
                      // multiple={false}
                      disabled={false}
                      reset={isReset}
                      activateFileName={true}
                      onFileUpload={handleAbbGtcFileUpload}
                      accept=".pdf,.docx"
                      format={true}
                      api_message={setApiMessage}
                    />
                  )}
                </div>

                <div className="numberOfSupplier">
                  <label
                    htmlFor="selectNumberSupplier"
                    className="labelHeading"
                  >
                    Select number of supplier(s):{" "}
                  </label>
                  <Dropdown
                    id="selectNumberSupplier"
                    className="dropdown"
                    value={numberOfSuppliers}
                    options={supplierOptions}
                    onChange={handleNumberOfSuppliersChange}
                    placeholder="Select Suppliers"
                    disabled={
                      !selectedPreloadAbbGTC || !isSupplierUploadEnabled
                    }
                    style={{ width: "150px", height: "50px" }}
                  />
                  {/* <select
                    id="selectNumberSupplier"
                    className="selectNumberOfSupplier"
                    disabled={
                      !selectedPreloadAbbGTC || !isSupplierUploadEnabled
                    }
                    value={numberOfSuppliers}
                    onChange={handleNumberOfSuppliersChange}
                  >
                    <option value="" />
                    {[...Array(1)].map((_, i) => (
                      <option key={i + 1} value={i + 1}>
                        {i + 1}
                      </option>
                    ))}
                  </select> */}
                </div>
                <div className="supplierFileUploadsBox">
                  <div className="supplierFileUploadsText labelHeading">
                    Upload {numberOfSuppliers} supplier(s) GTC:
                  </div>

                  {Array.from({ length: numberOfSuppliers }).map((_, index) => (
                    <div key={index} className="supplierFileUpload">
                      <BrowseFile
                        inputId={`supplierFileUpload_${index}`}
                        // multiple={false}
                        disabled={
                          !selectedPreloadAbbGTC || !isSupplierUploadEnabled
                        }
                        reset={isReset}
                        activateFileName={true}
                        onFileUpload={(file) =>
                          handleSupplierFileUpload(index, file)
                        }
                        accept=".pdf,.docx,.doc,.xls,.xlsx,.csv,.pptx"
                        format={false}
                        api_message={setApiMessage}
                      />
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="resetAndUploadButtonBox">
              <button className="resetButton" onClick={handleReset}>
                Reset
              </button>

              <button
                className="uploadButton"
                onClick={handleUpload}
                disabled={
                  (selectedPreloadAbbGTC == " " && abbGtcFileUpload == null) ||
                  !supplierFileUploads[0]
                }
              >
                Upload
              </button>
            </div>
          </>
        )}

        {isLoading && <Loader loaderContent="Analyzing..." />}

        <div ref={scrollBottomRef}></div>
      </div>
    </>
  );
};

export default GTC;
