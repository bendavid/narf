#include "ROOT/RDFHelpers.hxx"


#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <io.h>
#include <Windows.h>
#else
#include <sys/ioctl.h>
#endif


namespace narf {

// adapted from ROOT::RDF::Experimental::ProgressHelper
class ProgressHelper {
private:
   double EvtPerSec() const;
   std::pair<std::size_t, std::chrono::seconds> RecordEvtCountAndTime();
   void PrintStats(std::ostream &stream, std::size_t currentEventCount, std::chrono::seconds totalElapsedSeconds) const;
   void PrintStatsFinal(std::ostream &stream, std::chrono::seconds totalElapsedSeconds) const;
   void PrintProgressBar(std::ostream &stream, std::size_t currentEventCount) const;
   void PrintFileProgressBar(std::ostream &stream) const;

   std::chrono::time_point<std::chrono::system_clock> fBeginTime = std::chrono::system_clock::now();
   std::chrono::time_point<std::chrono::system_clock> fLastPrintTime = fBeginTime;
   std::chrono::seconds fPrintInterval{1};

   std::atomic<std::size_t> fProcessedEvents{0};
   std::size_t fLastProcessedEvents{0};
   std::size_t fIncrement;

   mutable std::mutex fSampleNameToEventEntriesMutex;
   std::map<std::string, ULong64_t> fSampleNameToEventEntries; // Filename, events in the file

   std::array<double, 20> fEventsPerSecondStatistics;
   std::size_t fEventsPerSecondStatisticsIndex{0};

   unsigned int fBarWidth;
   unsigned int fTotalFiles;

   std::mutex fPrintMutex;
   bool fIsTTY;
   bool fUseShellColours;

   std::shared_ptr<TTree> fTree{nullptr};

    struct RestoreStreamState {
        RestoreStreamState(std::ostream &stream) : fStream(stream), fFlags(stream.flags()), fFillChar(stream.fill()) {}
        ~RestoreStreamState()
        {
            fStream.setf(fFlags);
            fStream.fill(fFillChar);
        }

        std::ostream &fStream;
        std::ios_base::fmtflags fFlags;
        std::ostream::char_type fFillChar;
    };

public:
   /// Create a progress helper.
   /// \param increment RDF callbacks are called every `n` events. Pass this `n` here.
   /// \param totalFiles read total number of files in the RDF.
   /// \param progressBarWidth Number of characters the progress bar will occupy.
   /// \param printInterval Update every stats every `n` seconds.
   /// \param useColors Use shell colour codes to colour the output. Automatically disabled when
   /// we are not writing to a tty.
   ProgressHelper(std::size_t increment, unsigned int totalFiles = 1, unsigned int progressBarWidth = 40,
                  unsigned int printInterval = 1, bool useColors = true);

   ~ProgressHelper() = default;

   friend class ProgressBarAction;

   /// Register a new sample for completion statistics.
   /// \see ROOT::RDF::RInterface::DefinePerSample().
   /// The *id.AsString()* refers to the name of the currently processed file.
   /// The idea is to populate the  event entries in the *fSampleNameToEventEntries* map
   /// by selecting the greater of the two values:
   /// *id.EntryRange().second* which is the upper event entry range of the processed sample
   /// and the current value of the event entries in the *fSampleNameToEventEntries* map.
   /// In the single threaded case, the two numbers are the same as the entry range corresponds
   /// to the number of events in an individual file (each sample is simply a single file).
   /// In the multithreaded case, the idea is to accumulate the higher event entry value until
   /// the total number of events in a given file is reached.
   void registerNewSample(unsigned int /*slot*/, const ROOT::RDF::RSampleInfo &id)
   {
      std::lock_guard<std::mutex> lock(fSampleNameToEventEntriesMutex);
      fSampleNameToEventEntries[id.AsString()] =
         std::max(id.EntryRange().second, fSampleNameToEventEntries[id.AsString()]);
   }

   /// Thread-safe callback for RDataFrame.
   /// It will record elapsed times and event statistics, and print a progress bar every n seconds (set by the
   /// fPrintInterval). \param slot Ignored. \param value Ignored.
   template <typename T>
   void operator()(unsigned int /*slot*/, T &value)
   {
      operator()(value);
   }
   // clang-format off
   /// Thread-safe callback for RDataFrame.
   /// It will record elapsed times and event statistics, and print a progress bar every n seconds (set by the fPrintInterval).
   /// \param value Ignored.
   // clang-format on
   template <typename T>
   void operator()(T & /*value*/)
   {
      using namespace std::chrono;
      // ***************************************************
      // Warning: Here, everything needs to be thread safe:
      // ***************************************************
      fProcessedEvents += fIncrement;

      // We only print every n seconds.
      if (duration_cast<seconds>(system_clock::now() - fLastPrintTime) < fPrintInterval) {
         return;
      }

      // ***************************************************
      // Protected by lock from here:
      // ***************************************************
      if (!fPrintMutex.try_lock())
         return;
      std::lock_guard<std::mutex> lockGuard(fPrintMutex, std::adopt_lock);

      std::size_t eventCount;
      seconds elapsedSeconds;
      std::tie(eventCount, elapsedSeconds) = RecordEvtCountAndTime();

      if (fIsTTY)
        std::cout << "\r";

      PrintFileProgressBar(std::cout);
      PrintProgressBar(std::cout, eventCount);
      PrintStats(std::cout, eventCount, elapsedSeconds);


      if (fIsTTY)
        std::cout << std::flush;
      else
        std::cout << std::endl;

   }

   std::size_t ComputeNEventsSoFar() const
   {
      std::unique_lock<std::mutex> lock(fSampleNameToEventEntriesMutex);
      std::size_t result = 0;
      for (const auto &item : fSampleNameToEventEntries)
         result += item.second;
      return result;
   }

   unsigned int ComputeCurrentFileIdx() const
   {
      std::unique_lock<std::mutex> lock(fSampleNameToEventEntriesMutex);
      return fSampleNameToEventEntries.size();
   }

    // Get terminal size for progress bar
    int get_tty_size()
    {
        #if defined(_WIN32) || defined(_WIN64)
        if (!_isatty(_fileno(stdout)))
            return 0;
        int width = 0;
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi))
            width = (int)(csbi.srWindow.Right - csbi.srWindow.Left + 1);
        return width;
        #else
        int width = 0;
        struct winsize w;
        ioctl(fileno(stdout), TIOCGWINSZ, &w);
        width = (int)(w.ws_col);
        return width;
        #endif
    }

    void DoFinal() {
        const auto &[eventCount, elapsedSeconds] = RecordEvtCountAndTime();

        // The next line resets the current line output in the terminal.
        // Brings the cursor at the beginning ('\r'), prints whitespace with the
        // same length as the terminal size, then resets the cursor again so we
        // can print the final stats on a clean line.
        if (fIsTTY)
            std::cout << '\r' << std::string(get_tty_size(), ' ') << '\r';
        PrintStatsFinal(std::cout, elapsedSeconds);
        std::cout << std::endl;
    }

};

    // adapted from ROOT::RDF::Experimental::ProgressBarAction

    class ProgressBarAction final : public ROOT::Detail::RDF::RActionImpl<ProgressBarAction> {
    public:
    using Result_t = int;

    private:
    std::shared_ptr<ProgressHelper> fHelper;
    std::shared_ptr<int> fDummyResult = std::make_shared<int>();

    public:
    ProgressBarAction(std::shared_ptr<ProgressHelper> r) : fHelper(std::move(r)) {}

    std::shared_ptr<Result_t> GetResultPtr() const { return fDummyResult; }

    void Initialize() {}
    void InitTask(TTreeReader *, unsigned int) {}

    void Exec(unsigned int) {}

    void Finalize() {}

    std::string GetActionName() { return "ProgressBar"; }
    // dummy implementation of PartialUpdate
    int &PartialUpdate(unsigned int) { return *fDummyResult; }

    ROOT::RDF::SampleCallback_t GetSampleCallback() final
    {
        return [this](unsigned int slot, const ROOT::RDF::RSampleInfo &id) {
            this->fHelper->registerNewSample(slot, id);
            return this->fHelper->ComputeNEventsSoFar();
        };
    }
    };

    unsigned int RunGraphsWithProgressBar(std::vector<ROOT::RDF::RNode> nodes, unsigned int nentry=1000, unsigned int interval=1) {

        unsigned int total_files = 0;
        for (auto &node : nodes) {
            total_files += node.GetNFiles();
        }

        auto progress = std::make_shared<ProgressHelper>(nentry, total_files, 40, interval);

        std::vector<ROOT::RDF::RResultHandle> handles;
        handles.reserve(nodes.size());

        for (auto &node : nodes) {
            auto r = node.Book<>(ProgressBarAction(progress));
            auto rr = r.OnPartialResultSlot(nentry, [progress](unsigned int slot, auto &&arg) { (*progress)(slot, arg); });
            handles.emplace_back(std::move(rr));
        }

        const unsigned int res = ROOT::RDF::RunGraphs(handles);

        progress->DoFinal();

        return res;
    }

    unsigned int RunGraphsWithProgressBar(std::vector<ROOT::RDataFrame> dataframes, unsigned int nentry=1000, unsigned int interval=1) {

        std::vector<ROOT::RDF::RNode> nodes;
        nodes.reserve(dataframes.size());

        for (auto &dataframe : dataframes) {
            nodes.emplace_back(ROOT::RDF::AsRNode(dataframe));
        }

        return RunGraphsWithProgressBar(nodes, nentry, interval);
    }

ProgressHelper::ProgressHelper(std::size_t increment, unsigned int totalFiles, unsigned int progressBarWidth,
                               unsigned int printInterval, bool useColors)
   : fPrintInterval(printInterval),
     fIncrement{increment},
     fBarWidth{20},
     fTotalFiles{totalFiles},
#if defined(_WIN32) || defined(_WIN64)
     fIsTTY{_isatty(_fileno(stdout)) != 0},
     fUseShellColours{false && useColors}
#else
     fIsTTY{isatty(fileno(stdout)) == 1},
     fUseShellColours{useColors && fIsTTY} // Control characters only with terminals.
#endif
{
}

/// Compute a running mean of events/s.
double ProgressHelper::EvtPerSec() const
{
   if (fEventsPerSecondStatisticsIndex < fEventsPerSecondStatistics.size())
      return std::accumulate(fEventsPerSecondStatistics.begin(),
                             fEventsPerSecondStatistics.begin() + fEventsPerSecondStatisticsIndex, 0.) /
             fEventsPerSecondStatisticsIndex;
   else
      return std::accumulate(fEventsPerSecondStatistics.begin(), fEventsPerSecondStatistics.end(), 0.) /
             fEventsPerSecondStatistics.size();
}

/// Record current event counts and time stamp, populate evts/s statistics array.
std::pair<std::size_t, std::chrono::seconds> ProgressHelper::RecordEvtCountAndTime()
{
   using namespace std::chrono;

   auto currentEventCount = fProcessedEvents.load();
   auto eventsPerTimeInterval = currentEventCount - fLastProcessedEvents;
   fLastProcessedEvents = currentEventCount;

   auto oldPrintTime = fLastPrintTime;
   auto newPrintTime = system_clock::now();
   fLastPrintTime = newPrintTime;

   duration<double> secondsCurrentInterval = newPrintTime - oldPrintTime;
   fEventsPerSecondStatistics[fEventsPerSecondStatisticsIndex++ % fEventsPerSecondStatistics.size()] =
      eventsPerTimeInterval / secondsCurrentInterval.count();

   return {currentEventCount, duration_cast<seconds>(newPrintTime - fBeginTime)};
}

namespace {
/// Format std::chrono::seconds as `1:30m`.
std::ostream &operator<<(std::ostream &stream, std::chrono::seconds elapsedSeconds)
{
   auto h = std::chrono::duration_cast<std::chrono::hours>(elapsedSeconds);
   auto m = std::chrono::duration_cast<std::chrono::minutes>(elapsedSeconds - h);
   auto s = (elapsedSeconds - h - m).count();
   if (h.count() > 0)
      stream << h.count() << ':' << std::setw(2) << std::right << std::setfill('0');
   stream << m.count() << ':' << std::setw(2) << std::right << std::setfill('0') << s;
   return stream << (h.count() > 0 ? 'h' : 'm');
}


} // namespace

/// Print event and time statistics.
void ProgressHelper::PrintStats(std::ostream &stream, std::size_t currentEventCount,
                                std::chrono::seconds elapsedSeconds) const
{
   auto evtpersec = EvtPerSec();
   auto GetNEventsOfCurrentFile = ComputeNEventsSoFar();
   auto currentFileIdx = ComputeCurrentFileIdx();
   auto totalFiles = fTotalFiles;

   if (fUseShellColours)
      stream << "\033[35m";
   stream << "["
          << "Elapsed time: " << elapsedSeconds << "  ";
   if (fUseShellColours)
      stream << "\033[0m";
   stream << "processing file: " << currentFileIdx << " / " << totalFiles << "  ";

   // Event counts:
   if (fUseShellColours)
      stream << "\033[32m";

   stream << "processed evts: " << currentEventCount;
   if (GetNEventsOfCurrentFile != 0) {
      stream << " / " << std::scientific << std::setprecision(2) << GetNEventsOfCurrentFile;
   }
   stream << "  ";

   if (fUseShellColours)
      stream << "\033[0m";

   // events/s
   stream << std::scientific << std::setprecision(2) << evtpersec << " evt/s";

   // Time statistics:
   if (GetNEventsOfCurrentFile != 0) {
      if (fUseShellColours)
         stream << "\033[35m";
      std::chrono::seconds remainingSeconds(
         static_cast<long long>((ComputeNEventsSoFar() - currentEventCount) / evtpersec));
      // stream << " " << remainingSeconds << " "
             // << " remaining time (per file being processed)";
      if (fUseShellColours)
         stream << "\033[0m";
   }

   stream << "]   ";
}

void ProgressHelper::PrintStatsFinal(std::ostream &stream, std::chrono::seconds elapsedSeconds) const
{
   auto totalEvents = ComputeNEventsSoFar();
   auto totalFiles = fTotalFiles;

   if (fUseShellColours)
      stream << "\033[35m";
   stream << "["
          << "Total elapsed time: " << elapsedSeconds << "  ";
   if (fUseShellColours)
      stream << "\033[0m";
   stream << "processed files: " << totalFiles << " / " << totalFiles << "  ";

   // Event counts:
   if (fUseShellColours)
      stream << "\033[32m";

   stream << "processed evts: " << totalEvents;
   if (totalEvents != 0) {
      stream << " / " << std::scientific << std::setprecision(2) << totalEvents;
   }

   if (fUseShellColours)
      stream << "\033[0m";

   stream << "]   ";
}

/// Print a progress bar of width `ProgressHelper::fBarWidth` if `fGetNEventsOfCurrentFile` is known.
void ProgressHelper::PrintProgressBar(std::ostream &stream, std::size_t currentEventCount) const
{
   auto GetNEventsOfCurrentFile = ComputeNEventsSoFar();
   if (GetNEventsOfCurrentFile == 0)
      return;

   RestoreStreamState restore(stream);

   double completion = double(currentEventCount) / GetNEventsOfCurrentFile;
   unsigned int nBar = std::min(completion, 1.) * fBarWidth;

   std::string bars(std::max(nBar, 1u), '=');
   bars.back() = (nBar == fBarWidth) ? '=' : '>';

   if (fUseShellColours)
      stream << "\033[33m";
   stream << '|' << std::setfill(' ') << std::setw(fBarWidth) << std::left << bars << "|   ";
   if (fUseShellColours)
      stream << "\033[0m";
}

void ProgressHelper::PrintFileProgressBar(std::ostream &stream) const
{

    auto currentFileIdx = ComputeCurrentFileIdx();
    auto totalFiles = fTotalFiles;

    RestoreStreamState restore(stream);

    double completion = double(currentFileIdx) / totalFiles;
    unsigned int nBar = std::min(completion, 1.) * fBarWidth;

    std::string bars(std::max(nBar, 1u), '=');
    bars.back() = (nBar == fBarWidth) ? '=' : '>';

    if (fUseShellColours)
        stream << "\033[33m";
    stream << '|' << std::setfill(' ') << std::setw(fBarWidth) << std::left << bars << "|   ";
    if (fUseShellColours)
        stream << "\033[0m";
}


//*/


}


