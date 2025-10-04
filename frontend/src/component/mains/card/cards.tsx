// card.tsx
import React, { useState } from 'react';
import './cards.css'; // Create corresponding CSS file for styling
import ContractsIcon from 'assets/icon/ip-gtc-icon';
import CommercialIcon from 'assets/icon/ip-commercial-icon';
import { Size } from 'assets/icon/iconUtils';

interface CardsProps {
    onCardClick: (category: string) => void 
}

type CardInfo = {
    category: string;
    iconComponent: React.FC<{ className?: string; size?: Size }>;
}

const cardData: CardInfo[] = [
    // { category: "Single", iconName:"abb/document" },
    { category: "GTC", iconComponent: ContractsIcon },
    // { category: "Commercial", iconComponent: CommercialIcon },
];

const Cards: React.FC<CardsProps> = ({ onCardClick }) => {
    const [isCardClicked , setIsCardClicked] = useState<number | null>(null);

    const handleCategory = (cardIndex: number, category: string) => {
        setIsCardClicked(cardIndex);
        onCardClick(category); // Call the function passed from Welcome
    };

    return (
        <div className="cardMenu">
            {cardData.map((card, index) => (
                <div
                    key={index}
                    className={`card ${isCardClicked === index ? "clicked" : ""}`}
                    onClick={() => handleCategory(index, card.category)}
                >
                    <card.iconComponent
                        className="cardIcon"
                        size="large"
                    />
                    <h1> {card.category} <br /> Comparison</h1>
                </div>
            ))}
        </div>
    );
}

export default Cards;
