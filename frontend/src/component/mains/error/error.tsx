// error.tsx
import React from 'react';

const ErrorAnalyse: React.FC = () => {
    const error = 'Error: Cannot find module \'component/content/content.json\'';

    return (
        <div>
            <h1>Error</h1>
            <p>{error}</p>
        </div>
    )

};

export default ErrorAnalyse;