// Loader.tsx
import React from 'react';
import './loader.css'; 

interface LoaderProps {
    loaderContent?: string;
}

const Loader: React.FC<LoaderProps> = ({ loaderContent }) => {
    return (
        <div className="loaderBox">
            <div className="bouncingLoader">
                <div></div>
                <div></div>
                <div></div>
            </div>
            <h2>{loaderContent}</h2>
        </div>
    );
};

export default Loader;