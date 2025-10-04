// welcome.tsx
import React, { useState } from "react";
import "./welcome.css";
import WelcomeText from "component/mains/welcomeText/welcomeText";
import Footer from "component/mains/footer/footer";
import Header from "component/mains/header/header";
import Cards from "component/mains/card/cards";
import GTC from "component/mains/gtc/gtc";
import Comparison from "../comparison/comparison";
import CommercialCards from "component/mains/commercial/commercialCard";
import CommercialComp from "component/mains/commercial/commercialComp";

type DataList = {
  [entity: string]: {
    similarities: string[];
    additions: string[];
    removals: string[];
    differences: string;
  };
};

const Welcome: React.FC = () => {
  const [showCards, setShowCards] = useState<boolean>(true);
  const [selectedCard, setSelectedCard] = useState<string | null>(null);
  const [blobName, setBlobName] = useState<string>("");
  const [uploadCompleted, setUploadCompleted] = useState<boolean>(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [dataList, setDataList] = useState<DataList>({});
  const [isTokenCost, setIsTokenCost] = useState<number>(0);

  const handleCardClick = (category: string) => {
    setSelectedCard(category);
    setShowCards(false);
  };

  const handleBackClick = () => {
    setShowCards(true);
    setSelectedCard(null);
    setUploadCompleted(false);
  };

  const handleUploadClick = () => {
    setUploadCompleted(true);
  };

  const handleDataListUpdate = (newDataList: DataList) => {
    setDataList(newDataList);
    // Handle dataList update as needed
  };

  const handleDownloadClick = (blobName: string) => {
    setBlobName(blobName);
  };

  const handleTokenCost = (tokenCost: number) => {
    setIsTokenCost(tokenCost);
  };

  return (
    <div>
      <Header />
      <div className="welcome">
        <div className="welcomeAndCardBox">
          {(showCards || selectedCard === "GTC") && !uploadCompleted && (
            <WelcomeText isLoading={isLoading} />
          )}

          <div className="welcomeAndCardBox__card">
            {showCards && <Cards onCardClick={handleCardClick} />}
            {showCards && <CommercialCards onCardClick={handleCardClick} />}
          </div>

          {selectedCard === "GTC" && !uploadCompleted && (
            <GTC
              onBackClick={handleBackClick}
              onUploadClick={handleUploadClick}
              setIsLoader={setIsLoading}
              onDataListUpdate={handleDataListUpdate}
              onDownloadClick={handleDownloadClick}
              onTokenCost={handleTokenCost}
            />
          )}
          {selectedCard === "RFQ" && !uploadCompleted && (
            <CommercialComp
              onBackClick={handleBackClick}
              onUploadClick={handleUploadClick}
              setIsLoader={setIsLoading}
              onDataListUpdate={handleDataListUpdate}
              onDownloadClick={handleDownloadClick}
              onTokenCost={handleTokenCost}
            />
          )}
        </div>

        {uploadCompleted && (
          <Comparison
            onBackClick={handleBackClick}
            dataList={dataList}
            blobName={blobName}
            tokenCost={isTokenCost}
          />
        )}
      </div>
      <Footer />
    </div>
  );
};

export default Welcome;
